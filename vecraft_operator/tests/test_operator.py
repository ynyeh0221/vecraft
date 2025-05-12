import unittest
from unittest.mock import patch, MagicMock, call

import kubernetes
import vecraft_operator


class TestVecraftOperator(unittest.TestCase):

    def setUp(self):
        # Mock the Kubernetes configuration
        self.kube_config_patch = patch('kubernetes.config.load_incluster_config')
        self.mock_incluster_config = self.kube_config_patch.start()

        self.kube_local_config_patch = patch('kubernetes.config.load_kube_config')
        self.mock_kube_config = self.kube_local_config_patch.start()

        # Mock the Kubernetes API clients
        self.apps_api_patch = patch('kubernetes.client.AppsV1Api')
        self.mock_apps_api = self.apps_api_patch.start()
        self.core_api_patch = patch('kubernetes.client.CoreV1Api')
        self.mock_core_api = self.core_api_patch.start()

        # Create a mock for the kopf module and the owner_reference function
        self.kopf_patch = patch('vecraft_operator.kopf', autospec=True)
        self.mock_kopf = self.kopf_patch.start()

        # Configure the mock owner_reference function
        self.mock_owner_ref = MagicMock()
        self.mock_owner_ref.return_value = {"apiVersion": "test", "kind": "test", "name": "test", "uid": "test-uid"}
        self.mock_kopf.owner_reference = self.mock_owner_ref

        # Mock API instances that will be returned by the API constructors
        self.apps = MagicMock()
        self.core = MagicMock()
        self.mock_apps_api.return_value = self.apps
        self.mock_core_api.return_value = self.core

    def tearDown(self):
        # Stop all patches
        self.kube_config_patch.stop()
        self.kube_local_config_patch.stop()
        self.apps_api_patch.stop()
        self.core_api_patch.stop()
        self.kopf_patch.stop()

    def test_load_incluster_config_success(self):
        # Test that when load_incluster_config succeeds, load_kube_config is not called
        self.mock_incluster_config.side_effect = None

        # Reset the side effects
        self.mock_kube_config.reset_mock()

        # Import the module to apply our patches
        import importlib
        importlib.reload(vecraft_operator)

        self.mock_incluster_config.assert_called_once()
        self.mock_kube_config.assert_not_called()

    def test_load_incluster_config_fallback(self):
        # Test that when load_incluster_config fails, load_kube_config is called
        self.mock_incluster_config.side_effect = kubernetes.config.ConfigException("Test exception")

        # Reset the side effects
        self.mock_kube_config.reset_mock()

        # Import the module to apply our patches
        import importlib
        importlib.reload(vecraft_operator)

        self.mock_incluster_config.assert_called_once()
        self.mock_kube_config.assert_called_once()

    def test_create_vecraft(self):
        # Test create_vecraft handler
        logger = MagicMock()

        spec = {
            'root': '/data',
            'image': 'custom/image:latest',
            'replicas': 3
        }
        name = 'test-vecraft'
        namespace = 'test-namespace'

        # Call the handler
        result = vecraft_operator.create_vecraft(
            spec=spec, name=name, namespace=namespace, logger=logger, body=MagicMock()
        )

        # Check that kopf.owner_reference was called
        self.mock_owner_ref.assert_called()

        # Check that the correct service was created
        self.core.create_namespaced_service.assert_called_once()
        service_call = self.core.create_namespaced_service.call_args
        self.assertEqual(service_call[1]['namespace'], namespace)
        service_body = service_call[1]['body']
        self.assertEqual(service_body.metadata.name, name)
        self.assertEqual(service_body.metadata.namespace, namespace)
        self.assertEqual(service_body.spec.cluster_ip, 'None')  # Headless service
        self.assertEqual(service_body.spec.selector, {'app': 'vecraft', 'instance': name})
        self.assertEqual(len(service_body.spec.ports), 1)
        self.assertEqual(service_body.spec.ports[0].port, 80)
        self.assertEqual(service_body.spec.ports[0].target_port, 8000)

        # Check that the correct statefulset was created
        self.apps.create_namespaced_stateful_set.assert_called_once()
        sts_call = self.apps.create_namespaced_stateful_set.call_args
        self.assertEqual(sts_call[1]['namespace'], namespace)
        sts_body = sts_call[1]['body']
        self.assertEqual(sts_body.metadata.name, name)
        self.assertEqual(sts_body.metadata.namespace, namespace)
        self.assertEqual(sts_body.spec.replicas, 3)
        self.assertEqual(sts_body.spec.service_name, name)
        self.assertEqual(sts_body.spec.template.spec.containers[0].image, 'custom/image:latest')
        self.assertEqual(sts_body.spec.template.spec.containers[0].args, ['--root', '/data'])

        # Check that the function returned the expected result
        self.assertEqual(result, {'statefulset': name, 'service': name})

    def test_create_vecraft_defaults(self):
        # Test create_vecraft handler with default values
        logger = MagicMock()

        spec = {
            'root': '/data',
            # No image or replicas specified, should use defaults
        }
        name = 'test-vecraft'
        namespace = 'test-namespace'

        # Call the handler
        result = vecraft_operator.create_vecraft(
            spec=spec, name=name, namespace=namespace, logger=logger, body=MagicMock()
        )

        # Check that the statefulset was created with default values
        sts_call = self.apps.create_namespaced_stateful_set.call_args
        sts_body = sts_call[1]['body']
        self.assertEqual(sts_body.spec.replicas, 1)  # Default replicas
        self.assertEqual(sts_body.spec.template.spec.containers[0].image, 'myrepo/vecraft-rest:latest')  # Default image

    def test_update_vecraft_replicas(self):
        # Test update_vecraft handler with replicas change
        logger = MagicMock()

        spec = {
            'root': '/data',
            'replicas': 5,
            # No image change
        }
        name = 'test-vecraft'
        namespace = 'test-namespace'

        # Call the handler
        vecraft_operator.update_vecraft(
            spec=spec, status={}, name=name, namespace=namespace, logger=logger, body=MagicMock()
        )

        # Check that the statefulset was patched with the replicas change
        self.apps.patch_namespaced_stateful_set.assert_called_once()
        patch_call = self.apps.patch_namespaced_stateful_set.call_args
        self.assertEqual(patch_call[1]['name'], name)
        self.assertEqual(patch_call[1]['namespace'], namespace)
        self.assertEqual(patch_call[1]['body'], {'spec': {'replicas': 5}})

    def test_update_vecraft_image(self):
        # Test update_vecraft handler with image change
        logger = MagicMock()

        spec = {
            'root': '/data',
            'image': 'new/image:v2',
            # No replicas change
        }
        name = 'test-vecraft'
        namespace = 'test-namespace'

        # Call the handler
        vecraft_operator.update_vecraft(
            spec=spec, status={}, name=name, namespace=namespace, logger=logger, body=MagicMock()
        )

        # Check that the statefulset was patched with the image change
        self.apps.patch_namespaced_stateful_set.assert_called_once()
        patch_call = self.apps.patch_namespaced_stateful_set.call_args
        self.assertEqual(patch_call[1]['name'], name)
        self.assertEqual(patch_call[1]['namespace'], namespace)
        self.assertEqual(
            patch_call[1]['body'],
            {'spec': {'template': {'spec': {'containers': [{'name': 'vecraft', 'image': 'new/image:v2'}]}}}}
        )

    def test_update_vecraft_both(self):
        # Test update_vecraft handler with both replicas and image changes
        logger = MagicMock()

        spec = {
            'root': '/data',
            'replicas': 2,
            'image': 'new/image:v2',
        }
        name = 'test-vecraft'
        namespace = 'test-namespace'

        # Call the handler
        vecraft_operator.update_vecraft(
            spec=spec, status={}, name=name, namespace=namespace, logger=logger, body=MagicMock()
        )

        # Check that the statefulset was patched with both changes
        self.apps.patch_namespaced_stateful_set.assert_called_once()
        patch_call = self.apps.patch_namespaced_stateful_set.call_args
        self.assertEqual(patch_call[1]['name'], name)
        self.assertEqual(patch_call[1]['namespace'], namespace)
        expected_patch = {
            'spec': {
                'replicas': 2,
                'template': {
                    'spec': {
                        'containers': [
                            {'name': 'vecraft', 'image': 'new/image:v2'}
                        ]
                    }
                }
            }
        }
        self.assertEqual(patch_call[1]['body'], expected_patch)

    def test_update_vecraft_no_changes(self):
        # Test update_vecraft handler with no changes
        logger = MagicMock()

        spec = {
            'root': '/data',
            # No replicas or image changes
        }
        name = 'test-vecraft'
        namespace = 'test-namespace'

        # Call the handler
        vecraft_operator.update_vecraft(
            spec=spec, status={}, name=name, namespace=namespace, logger=logger, body=MagicMock()
        )

        # Check that no patch was applied
        self.apps.patch_namespaced_stateful_set.assert_not_called()

    def test_delete_vecraft(self):
        # Test delete_vecraft handler
        logger = MagicMock()

        spec = {'root': '/data'}
        name = 'test-vecraft'
        namespace = 'test-namespace'

        # Call the handler
        vecraft_operator.delete_vecraft(
            spec=spec, name=name, namespace=namespace, logger=logger, body=MagicMock()
        )

        # Check that logger.info was called
        logger.info.assert_called_once()

        # Verify that the message contains the resource name
        log_message = logger.info.call_args[0][0]
        self.assertIn(name, log_message)


if __name__ == '__main__':
    unittest.main()