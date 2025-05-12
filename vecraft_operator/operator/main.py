import kopf
import kubernetes.client as k8s
from kubernetes.client import V1ObjectMeta, V1Service, V1ServiceSpec, V1ServicePort, V1Container, V1VolumeMount, \
    V1ResourceRequirements, V1PodSpec, V1PersistentVolumeClaim, V1PersistentVolumeClaimSpec, V1OwnerReference, \
    V1LabelSelector, V1PodTemplateSpec, V1StatefulSetSpec, V1StatefulSet
from kubernetes.client.rest import ApiException

# Constants
GROUP = "vecraft.io"
VERSION = "v1alpha1"
PLURAL = "vecraftclusters"

# Utility to create owner references

def owner_ref(body):
    return V1OwnerReference(
        api_version=f"{GROUP}/{VERSION}",
        kind=body['kind'],
        name=body['metadata']['name'],
        uid=body['metadata']['uid'],
        controller=True,
        block_owner_deletion=True,
    )

@kopf.on.create(GROUP, VERSION, PLURAL)
@kopf.on.update(GROUP, VERSION, PLURAL)
def reconcile_cluster(spec, status, name, namespace, body, **kwargs):
    api = k8s.CoreV1Api()
    apps = k8s.AppsV1Api()

    # 1) Headless Service
    headless_service = V1Service(
        metadata=V1ObjectMeta(
            name=f"{name}-headless",
            namespace=namespace,
            owner_references=[owner_ref(body)]
        ),
        spec=V1ServiceSpec(
            cluster_ip="None",
            selector={"app": "vecraft", "cluster": name},
            ports=[V1ServicePort(port=8000, target_port=8000)]
        )
    )
    try:
        api.create_namespaced_service(namespace, headless_service)
    except ApiException as e:
        if e.status != 409:
            raise

    # 2) StatefulSet
    ss_labels = {"app": "vecraft", "cluster": name}
    pvc = V1PersistentVolumeClaim(
        metadata=V1ObjectMeta(name="data"),
        spec=V1PersistentVolumeClaimSpec(
            access_modes=["ReadWriteOnce"],
            resources=V1ResourceRequirements(requests={"storage": spec.get("storage", {}).get("size", "10Gi")}),
            storage_class_name=spec.get("storage", {}).get("storageClassName")
        )
    )
    container = V1Container(
        name="vecraft",
        image=spec['image'],
        ports=[k8s.V1ContainerPort(container_port=8000)],
        volume_mounts=[V1VolumeMount(name="data", mount_path="/data")],
        resources=V1ResourceRequirements(**spec.get("resources", {}))
    )
    pod_spec = V1PodSpec(
        init_containers=[
            V1Container(
                name="fix-perms",
                image="busybox",
                command=["sh", "-c", "chmod 700 /data"],
                volume_mounts=[V1VolumeMount(name="data", mount_path="/data")]
            )
        ],
        containers=[container]
    )
    pod_template = V1PodTemplateSpec(
        metadata=V1ObjectMeta(labels=ss_labels),
        spec=pod_spec
    )
    stateful_set = V1StatefulSet(
        metadata=V1ObjectMeta(
            name=f"{name}-sts",
            namespace=namespace,
            owner_references=[owner_ref(body)]
        ),
        spec=V1StatefulSetSpec(
            service_name=f"{name}-headless",
            replicas=spec.get('replicas', 1),
            selector=V1LabelSelector(match_labels=ss_labels),
            template=pod_template,
            volume_claim_templates=[pvc]
        )
    )
    try:
        apps.create_namespaced_stateful_set(namespace, stateful_set)
    except ApiException as e:
        if e.status != 409:
            raise

    # 3) External Service
    external_svc = V1Service(
        metadata=V1ObjectMeta(
            name=f"{name}-svc",
            namespace=namespace,
            owner_references=[owner_ref(body)]
        ),
        spec=V1ServiceSpec(
            type="LoadBalancer",
            selector=ss_labels,
            ports=[V1ServicePort(port=8000, target_port=8000)]
        )
    )
    try:
        api.create_namespaced_service(namespace, external_svc)
    except ApiException as e:
        if e.status != 409:
            raise

    # 4) Update Status
    # fetch current resources
    sts = apps.read_namespaced_stateful_set(f"{name}-sts", namespace)
    svc = api.read_namespaced_service(f"{name}-svc", namespace)
    ready = sts.status.ready_replicas or 0
    phase = "Running" if ready >= spec.get('replicas', 1) else "Pending"
    endpoint = None
    if svc.status.load_balancer and svc.status.load_balancer.ingress:
        endpoint = svc.status.load_balancer.ingress[0].hostname or svc.status.load_balancer.ingress[0].ip

    status_patch = {
        "status": {
            "phase": phase,
            "readyReplicas": ready,
            "endpoint": endpoint,
        }
    }
    # Patch status subresource
    custom = k8s.CustomObjectsApi()
    custom.patch_namespaced_custom_object_status(
        GROUP, VERSION, namespace, PLURAL, name, status_patch
    )

@kopf.on.delete(GROUP, VERSION, PLURAL)
def delete_cluster(spec, name, namespace, **kwargs):
    # Resources have ownerReferences; GC will clean them up.
    kopf.info(f"VecraftCluster {name} deleted: children will be garbage-collected.")
