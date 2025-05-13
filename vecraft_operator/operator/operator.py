import kopf
import kubernetes
from kubernetes.client import AppsV1Api, CoreV1Api
from kubernetes.client import (
    V1StatefulSet, V1StatefulSetSpec, V1PodTemplateSpec,
    V1ObjectMeta, V1PodSpec, V1Container, V1Service,
    V1ServiceSpec, V1ServicePort, V1LabelSelector,
)

# Load in-cluster config (or fallback to local kubeconfig)
try:
    kubernetes.config.load_incluster_config()
except kubernetes.config.ConfigException:
    kubernetes.config.load_kube_config()

apps = AppsV1Api()
core = CoreV1Api()

@kopf.on.create('vecraft.io', 'v1alpha1', 'vecraftdatabases')
@kopf.on.resume('vecraft.io', 'v1alpha1', 'vecraftdatabases')
def reconcile_vecraft(spec, name, namespace, logger, **kwargs):
    """
    Ensures that the headless Service and StatefulSet match the CR spec.
    This handler runs on creation and on operator resume, using a patch (force)
    for true create-or-update semantics.
    """
    root     = spec['root']
    image    = spec.get('image', 'ghcr.io/ynyeh0221/vecraft:main')
    replicas = spec.get('replicas', 1)
    labels   = {'app': 'vecraft', 'instance': name}

    # Build headless Service manifest
    svc = V1Service(
        metadata=V1ObjectMeta(
            name=name,
            namespace=namespace,
            owner_references=[kopf.owner_reference(**kwargs)],
        ),
        spec=V1ServiceSpec(
            cluster_ip='None',
            selector=labels,
            ports=[V1ServicePort(port=80, target_port=8000)],
        )
    )

    # Patch (create-or-update) the Service
    core.patch_namespaced_service(
        name=name,
        namespace=namespace,
        body=svc,
        field_manager='vecraft-operator',
        force=True,
    )
    logger.info(f"Ensured headless Service '{name}'")

    # Build StatefulSet manifest
    sts = V1StatefulSet(
        metadata=V1ObjectMeta(
            name=name,
            namespace=namespace,
            owner_references=[kopf.owner_reference(**kwargs)],
        ),
        spec=V1StatefulSetSpec(
            replicas=replicas,
            selector=V1LabelSelector(match_labels=labels),
            service_name=name,
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(labels=labels),
                spec=V1PodSpec(containers=[
                    V1Container(
                        name='vecraft',
                        image=image,
                        args=['--root', root],
                        ports=[{'containerPort': 8000}],
                    )
                ])
            )
        )
    )

    # Patch (create-or-update) the StatefulSet
    apps.patch_namespaced_stateful_set(
        name=name,
        namespace=namespace,
        body=sts,
        field_manager='vecraft-operator',
        force=True,
    )
    logger.info(f"Ensured StatefulSet '{name}'")

    # Return some status info (optional)
    return {'statefulset': name, 'service': name}


@kopf.on.update('vecraft.io', 'v1alpha1', 'vecraftdatabases')
def on_update(spec, status, name, namespace, logger, **kwargs):
    """
    On spec changes, make sure we reconcile again.
    You can pump the update into the same reconcile logic:
    """
    # Simply call the reconcile function to re-patch with new spec
    return reconcile_vecraft(spec, name, namespace, logger, **kwargs)


@kopf.on.delete('vecraft.io', 'v1alpha1', 'vecraftdatabases')
def on_delete(spec, name, namespace, logger, **kwargs):
    """
    No deletion logic needed—ownerReferences will garbage-collect the
    Service and StatefulSet.
    """
    logger.info(f"VecraftDatabase '{name}' deleted; resources will be cleaned up by Kubernetes.")
