import unittest
from unittest.mock import patch, MagicMock

# Prevent Kubernetes config loading during import
import kubernetes.config
kubernetes.config.load_incluster_config = lambda: None
kubernetes.config.load_kube_config = lambda: None

from kubernetes.client import V1Service, V1StatefulSet

from vecraft_operator.operator.operator import (
    reconcile_vecraft,
    on_update,
    on_delete,
    core,
    apps,
)

class TestVecraftOperator(unittest.TestCase):
    def setUp(self):
        self.name = 'test-db'
        self.namespace = 'default'
        self.spec = {'root': '/data', 'replicas': 2, 'image': 'test/image:latest'}
        self.logger = MagicMock()
        self.owner_ref = {'uid': 'dummy-uid'}
        # Kopf.owner_reference is called with **kwargs, so we set body.metadata.uid here
        self.kwargs = {'body': {'metadata': {'uid': 'dummy-uid'}}}

    @patch('vecraft_operator.operator.operator.kopf.owner_reference', create=True)
    @patch.object(core, 'patch_namespaced_service')
    @patch.object(apps, 'patch_namespaced_stateful_set')
    def test_reconcile_vecraft(self, mock_patch_sts, mock_patch_svc, mock_owner_ref):
        # owner_reference should return our dummy
        mock_owner_ref.return_value = self.owner_ref

        # Call the reconcile (create & resume) handler
        result = reconcile_vecraft(
            spec=self.spec,
            name=self.name,
            namespace=self.namespace,
            logger=self.logger,
            **self.kwargs
        )

        # Service: patch_namespaced_service called once
        mock_patch_svc.assert_called_once()
        _, svc_kwargs = mock_patch_svc.call_args
        self.assertEqual(svc_kwargs['name'], self.name)
        self.assertEqual(svc_kwargs['namespace'], self.namespace)
        svc_body = svc_kwargs['body']
        self.assertIsInstance(svc_body, V1Service)
        self.assertEqual(svc_body.metadata.name, self.name)
        self.assertEqual(svc_body.spec.cluster_ip, 'None')
        self.assertEqual(svc_body.spec.selector, {'app': 'vecraft', 'instance': self.name})
        self.assertEqual(svc_kwargs['field_manager'], 'vecraft-operator')
        self.assertTrue(svc_kwargs['force'])
        self.logger.info.assert_any_call(f"Ensured headless Service '{self.name}'")

        # StatefulSet: patch_namespaced_stateful_set called once
        mock_patch_sts.assert_called_once()
        _, sts_kwargs = mock_patch_sts.call_args
        self.assertEqual(sts_kwargs['name'], self.name)
        self.assertEqual(sts_kwargs['namespace'], self.namespace)
        sts_body = sts_kwargs['body']
        self.assertIsInstance(sts_body, V1StatefulSet)
        self.assertEqual(sts_body.metadata.name, self.name)
        self.assertEqual(sts_kwargs['field_manager'], 'vecraft-operator')
        self.assertTrue(sts_kwargs['force'])
        self.logger.info.assert_any_call(f"Ensured StatefulSet '{self.name}'")

        # Return value
        self.assertEqual(result, {'statefulset': self.name, 'service': self.name})

    @patch('vecraft_operator.operator.operator.kopf.owner_reference', create=True)
    @patch.object(core, 'patch_namespaced_service')
    @patch.object(apps, 'patch_namespaced_stateful_set')
    def test_on_update_delegates_to_reconcile(self, mock_patch_sts, mock_patch_svc, mock_owner_ref):
        # Make sure on_update simply calls reconcile_vecraft under the covers
        mock_owner_ref.return_value = self.owner_ref

        on_update(
            spec=self.spec,
            status=None,
            name=self.name,
            namespace=self.namespace,
            logger=self.logger,
            **self.kwargs
        )

        # Both resources should be patched again
        mock_patch_svc.assert_called()
        mock_patch_sts.assert_called()

    def test_on_delete_logs_cleanup(self):
        on_delete(
            spec=self.spec,
            name=self.name,
            namespace=self.namespace,
            logger=self.logger
        )
        self.logger.info.assert_called_once_with(
            f"VecraftDatabase '{self.name}' deleted; resources will be cleaned up by Kubernetes."
        )

if __name__ == '__main__':
    unittest.main()
