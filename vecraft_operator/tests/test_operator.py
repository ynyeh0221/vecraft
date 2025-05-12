import unittest
from unittest.mock import patch, MagicMock

# Prevent Kubernetes config loading during import
import kubernetes.config
kubernetes.config.load_incluster_config = lambda: None
kubernetes.config.load_kube_config = lambda: None

from kubernetes.client import V1Service, V1StatefulSet

from vecraft_operator.operator.operator import (
    create_vecraft,
    update_vecraft,
    delete_vecraft,
    core,
    apps
)

class TestVecraftOperator(unittest.TestCase):
    def setUp(self):
        self.name = 'test-db'
        self.namespace = 'default'
        self.spec = {'root': '/data', 'replicas': 2, 'image': 'test/image:latest'}
        self.logger = MagicMock()
        self.owner_ref = {'uid': 'dummy-uid'}
        self.kwargs = {'body': {'metadata': {'uid': 'dummy-uid'}}}

    @patch('vecraft_operator.operator.operator.kopf.owner_reference', create=True)
    @patch.object(core, 'create_namespaced_service')
    @patch.object(apps, 'create_namespaced_stateful_set')
    def test_create_vecraft(self, mock_create_ss, mock_create_svc, mock_owner_ref):
        # Mock owner_reference to return a dummy reference
        mock_owner_ref.return_value = self.owner_ref

        # Call the creation handler
        result = create_vecraft(
            spec=self.spec,
            name=self.name,
            namespace=self.namespace,
            logger=self.logger,
            **self.kwargs
        )

        # Assert headless Service creation
        mock_create_svc.assert_called_once()
        svc_call = mock_create_svc.call_args[1]
        self.assertEqual(svc_call['namespace'], self.namespace)
        svc_body = svc_call['body']
        self.assertIsInstance(svc_body, V1Service)
        self.assertEqual(svc_body.metadata.name, self.name)
        self.assertEqual(svc_body.spec.cluster_ip, 'None')
        self.assertEqual(svc_body.spec.selector, {'app': 'vecraft', 'instance': self.name})
        self.logger.info.assert_any_call(f"Created headless Service {self.name}")

        # Assert StatefulSet creation
        mock_create_ss.assert_called_once()
        sts_call = mock_create_ss.call_args[1]
        self.assertEqual(sts_call['namespace'], self.namespace)
        sts_body = sts_call['body']
        self.assertIsInstance(sts_body, V1StatefulSet)
        self.assertEqual(sts_body.metadata.name, self.name)
        self.logger.info.assert_any_call(f"Created StatefulSet {self.name}")

        # Verify return value
        self.assertEqual(result, {'statefulset': self.name, 'service': self.name})

    @patch.object(apps, 'patch_namespaced_stateful_set')
    def test_update_vecraft_no_changes(self, mock_patch):
        # No spec changes should result in no patch call
        update_vecraft(
            spec={}, status=None,
            name=self.name, namespace=self.namespace,
            logger=self.logger
        )
        mock_patch.assert_not_called()

    @patch.object(apps, 'patch_namespaced_stateful_set')
    def test_update_vecraft_replicas_only(self, mock_patch):
        spec = {'replicas': 5}
        update_vecraft(
            spec=spec, status=None,
            name=self.name, namespace=self.namespace,
            logger=self.logger
        )
        mock_patch.assert_called_once_with(
            name=self.name,
            namespace=self.namespace,
            body={'spec': {'replicas': 5}}
        )
        self.logger.info.assert_called_with(f"Updated StatefulSet {self.name}")

    @patch.object(apps, 'patch_namespaced_stateful_set')
    def test_update_vecraft_image_only(self, mock_patch):
        spec = {'image': 'new/image:tag'}
        update_vecraft(
            spec=spec, status=None,
            name=self.name, namespace=self.namespace,
            logger=self.logger
        )
        mock_patch.assert_called_once_with(
            name=self.name,
            namespace=self.namespace,
            body={'spec': {'template': {'spec': {'containers': [
                {'name': 'vecraft', 'image': 'new/image:tag'}
            ]}}}}
        )
        self.logger.info.assert_called_with(f"Updated StatefulSet {self.name}")

    @patch.object(apps, 'patch_namespaced_stateful_set')
    def test_update_vecraft_both(self, mock_patch):
        spec = {'replicas': 3, 'image': 'another/image:tag'}
        update_vecraft(
            spec=spec, status=None,
            name=self.name, namespace=self.namespace,
            logger=self.logger
        )
        mock_patch.assert_called_once_with(
            name=self.name,
            namespace=self.namespace,
            body={
                'spec': {
                    'replicas': 3,
                    'template': {'spec': {'containers': [
                        {'name': 'vecraft', 'image': 'another/image:tag'}
                    ]}}
                }
            }
        )
        self.logger.info.assert_called_with(f"Updated StatefulSet {self.name}")

    def test_delete_vecraft_logs(self):
        delete_vecraft(
            spec=self.spec,
            name=self.name,
            namespace=self.namespace,
            logger=self.logger
        )
        self.logger.info.assert_called_with(
            f"VecraftDatabase {self.name} deleted; resources will be garbage-collected."
        )

if __name__ == '__main__':
    unittest.main()