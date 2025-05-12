import kopf
import kubernetes
from kubernetes.client import AppsV1Api, CoreV1Api
from kubernetes.client import (V1StatefulSet, V1StatefulSetSpec, V1PodTemplateSpec,
                                V1ObjectMeta, V1PodSpec, V1Container, V1Service,
                                V1ServiceSpec, V1ServicePort, V1LabelSelector)

# Load in-cluster config (or fallback to local kubeconfig)
try:
    kubernetes.config.load_incluster_config()
except kubernetes.config.ConfigException:
    kubernetes.config.load_kube_config()

apps = AppsV1Api()
core = CoreV1Api()

@kopf.on.create('vecraft.io', 'v1alpha1', 'vecraftdatabases')
def create_vecraft(spec, name, namespace, logger, **kwargs):
    root = spec['root']
    image = spec.get('image', 'ghcr.io/ynyeh0221/vecraft:main')
    replicas = spec.get('replicas', 1)
    labels = {'app': 'vecraft', 'instance': name}

    # Headless Service for StatefulSet
    headless = V1Service(
        metadata=V1ObjectMeta(name=name, namespace=namespace,
                              owner_references=[kopf.owner_reference(**kwargs)]),
        spec=V1ServiceSpec(
            cluster_ip='None',
            selector=labels,
            ports=[V1ServicePort(port=80, target_port=8000)]
        )
    )
    core.create_namespaced_service(namespace=namespace, body=headless)
    logger.info(f"Created headless Service {name}")

    # Create StatefulSet
    sts = V1StatefulSet(
        metadata=V1ObjectMeta(name=name, namespace=namespace,
                              owner_references=[kopf.owner_reference(**kwargs)]),
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
                        ports=[{'containerPort': 8000}]
                    )
                ])
            )
        )
    )
    apps.create_namespaced_stateful_set(namespace=namespace, body=sts)
    logger.info(f"Created StatefulSet {name}")

    return {'statefulset': name, 'service': name}

@kopf.on.update('vecraft.io', 'v1alpha1', 'vecraftdatabases')
def update_vecraft(spec, status, name, namespace, logger, **kwargs):
    replicas = spec.get('replicas')
    image = spec.get('image')
    patch = {'spec': {}}
    if replicas is not None:
        patch['spec']['replicas'] = replicas
    if image:
        patch['spec']['template'] = {
            'spec': {'containers': [{'name': 'vecraft', 'image': image}]}
        }
    if patch['spec']:
        apps.patch_namespaced_stateful_set(name=name, namespace=namespace, body=patch)
        logger.info(f"Updated StatefulSet {name}")

@kopf.on.delete('vecraft.io', 'v1alpha1', 'vecraftdatabases')
def delete_vecraft(spec, name, namespace, logger, **kwargs):
    logger.info(f"VecraftDatabase {name} deleted; resources will be garbage-collected.")