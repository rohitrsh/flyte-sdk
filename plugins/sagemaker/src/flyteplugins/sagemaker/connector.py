from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Sequence

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from flyte.connectors import AsyncConnector, ConnectorRegistry, Resource, ResourceMeta
from flyteidl2.connector.connector_pb2 import TaskExecutionMetadata
from flyteidl2.core.execution_pb2 import TaskExecution, TaskLog
from flyteidl2.core.tasks_pb2 import TaskTemplate
from google.protobuf.json_format import MessageToDict

from flyteplugins.sagemaker.task import (
    BOTO_TASK_TYPE,
    PROCESSING_TASK_TYPE,
    TRAINING_TASK_TYPE,
    TRANSFORM_TASK_TYPE,
)

_BOTO_CONFIG = Config(retries={"max_attempts": 3, "mode": "adaptive"})
_RUNNING_STATES = {"Starting", "InProgress", "Stopping"}
_RETRYABLE_ERROR_CODES = {
    "InternalFailure",
    "InternalServerException",
    "ServiceUnavailable",
    "ThrottlingException",
    "TooManyRequestsException",
}
_SERVICE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_METHOD_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _restore_integer_values(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_restore_integer_values(item) for item in value]
    if isinstance(value, dict):
        return {key: _restore_integer_values(item) for key, item in value.items()}
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return value


def _replace_idempotence_token(value: Any, token: str) -> Any:
    if isinstance(value, str):
        return value.replace("{idempotence_token}", token)
    if isinstance(value, list):
        return [_replace_idempotence_token(item, token) for item in value]
    if isinstance(value, dict):
        return {key: _replace_idempotence_token(item, token) for key, item in value.items()}
    return value


def _execution_token(
    metadata: Optional[TaskExecutionMetadata],
    custom: Dict[str, Any],
) -> str:
    if metadata is not None and metadata.HasField("task_execution_id"):
        source = metadata.task_execution_id.SerializeToString()
    else:
        source = json.dumps(
            custom,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    return hashlib.sha256(source).hexdigest()[:32]


@dataclass(frozen=True)
class _JobDefinition:
    task_type: str
    create_method: str
    describe_method: str
    stop_method: str
    name_key: str
    status_key: str
    failure_key: str
    console_path: str
    result_fields: Dict[str, str]


_TRAINING = _JobDefinition(
    task_type=TRAINING_TASK_TYPE,
    create_method="create_training_job",
    describe_method="describe_training_job",
    stop_method="stop_training_job",
    name_key="TrainingJobName",
    status_key="TrainingJobStatus",
    failure_key="FailureReason",
    console_path="jobs",
    result_fields={
        "ModelArtifacts": "model_artifacts",
        "OutputDataConfig": "output_data_config",
        "TrainingTimeInSeconds": "training_time_seconds",
        "BillableTimeInSeconds": "billable_time_seconds",
    },
)
_PROCESSING = _JobDefinition(
    task_type=PROCESSING_TASK_TYPE,
    create_method="create_processing_job",
    describe_method="describe_processing_job",
    stop_method="stop_processing_job",
    name_key="ProcessingJobName",
    status_key="ProcessingJobStatus",
    failure_key="FailureReason",
    console_path="processing-jobs",
    result_fields={
        "ProcessingOutputConfig": "processing_output_config",
        "ProcessingResources": "processing_resources",
    },
)
_TRANSFORM = _JobDefinition(
    task_type=TRANSFORM_TASK_TYPE,
    create_method="create_transform_job",
    describe_method="describe_transform_job",
    stop_method="stop_transform_job",
    name_key="TransformJobName",
    status_key="TransformJobStatus",
    failure_key="FailureReason",
    console_path="transform-jobs",
    result_fields={
        "TransformOutput": "transform_output",
        "TransformResources": "transform_resources",
    },
)


@dataclass
class SageMakerJobMetadata(ResourceMeta):
    region: Optional[str]
    job_name: str
    output_key: Optional[str] = None


class SageMakerManagedJobConnector(AsyncConnector[SageMakerJobMetadata]):
    definition: _JobDefinition
    metadata_type = SageMakerJobMetadata

    def _get_client(self, region: Optional[str]) -> Any:
        kwargs: Dict[str, Any] = {
            "service_name": "sagemaker",
            "config": _BOTO_CONFIG,
        }
        if region:
            kwargs["region_name"] = region
        return boto3.client(**kwargs)

    async def _call(self, client: Any, method: str, **request: Any) -> Dict[str, Any]:
        return await asyncio.to_thread(getattr(client, method), **request)

    def _output_key(self, task_template: TaskTemplate) -> Optional[str]:
        outputs = [variable.key for variable in task_template.interface.outputs.variables]
        if len(outputs) > 1:
            raise ValueError("SageMaker managed jobs support at most one Flyte output")
        return outputs[0] if outputs else None

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SageMakerJobMetadata:
        custom = MessageToDict(task_template.custom) if task_template.custom else {}
        request = _restore_integer_values(custom.get("request") or {})
        region = custom.get("region") or None
        job_name = request.get(self.definition.name_key)
        if not job_name:
            raise ValueError(f"SageMaker request.{self.definition.name_key} is required")

        client = self._get_client(region)
        try:
            await self._call(client, self.definition.create_method, **request)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code not in {"ConflictException", "ResourceInUse"}:
                raise
            await self._call(
                client,
                self.definition.describe_method,
                **{self.definition.name_key: job_name},
            )

        return SageMakerJobMetadata(
            region=region,
            job_name=job_name,
            output_key=self._output_key(task_template),
        )

    async def get(self, resource_meta: SageMakerJobMetadata, **kwargs: Any) -> Resource:
        client = self._get_client(resource_meta.region)
        try:
            response = await self._call(
                client,
                self.definition.describe_method,
                **{self.definition.name_key: resource_meta.job_name},
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            phase = TaskExecution.RETRYABLE_FAILED if code in _RETRYABLE_ERROR_CODES else TaskExecution.FAILED
            return Resource(phase=phase, message=str(error))

        state = response.get(self.definition.status_key, "")
        if state in _RUNNING_STATES:
            phase = TaskExecution.RUNNING
        elif state == "Completed":
            phase = TaskExecution.SUCCEEDED
        elif state == "Stopped":
            phase = TaskExecution.ABORTED
        elif state == "Failed":
            phase = TaskExecution.FAILED
        else:
            return Resource(
                phase=TaskExecution.FAILED,
                message=f"Unrecognized SageMaker job state {state!r}",
            )

        region = resource_meta.region or getattr(client.meta, "region_name", None) or "us-east-1"
        job_type = self.definition.task_type.removeprefix("sagemaker-").removesuffix("-job").title()
        log = TaskLog(
            uri=(
                f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}"
                f"#/{self.definition.console_path}/{resource_meta.job_name}"
            ),
            name=f"SageMaker {job_type} Console",
            ready=True,
            link_type=TaskLog.EXTERNAL,
        )
        message = state
        failure = response.get(self.definition.failure_key)
        if failure:
            message = f"{state}: {failure}"

        outputs: Optional[Dict[str, Any]] = None
        if phase == TaskExecution.SUCCEEDED:
            result: Dict[str, Any] = {
                "job_name": resource_meta.job_name,
                "status": state,
            }
            for aws_key, output_key in self.definition.result_fields.items():
                if aws_key in response:
                    result[output_key] = _json_safe(response[aws_key])
            outputs = {resource_meta.output_key: result} if resource_meta.output_key else {}

        return Resource(
            phase=phase,
            message=message,
            log_links=[log],
            outputs=outputs,
            custom_info={"job_name": resource_meta.job_name},
        )

    async def delete(self, resource_meta: SageMakerJobMetadata, **kwargs: Any) -> None:
        client = self._get_client(resource_meta.region)
        try:
            response = await self._call(
                client,
                self.definition.describe_method,
                **{self.definition.name_key: resource_meta.job_name},
            )
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code in {"ResourceNotFound", "ValidationException"}:
                return
            raise
        if response.get(self.definition.status_key) in {"Completed", "Failed", "Stopped"}:
            return
        await self._call(
            client,
            self.definition.stop_method,
            **{self.definition.name_key: resource_meta.job_name},
        )


class SageMakerTrainingConnector(SageMakerManagedJobConnector):
    name = "SageMaker Training Connector"
    task_type_name = TRAINING_TASK_TYPE
    definition = _TRAINING


class SageMakerProcessingConnector(SageMakerManagedJobConnector):
    name = "SageMaker Processing Connector"
    task_type_name = PROCESSING_TASK_TYPE
    definition = _PROCESSING


class SageMakerTransformConnector(SageMakerManagedJobConnector):
    name = "SageMaker Transform Connector"
    task_type_name = TRANSFORM_TASK_TYPE
    definition = _TRANSFORM


@dataclass
class BotoCallMetadata(ResourceMeta):
    response: Dict[str, Any]
    output_key: Optional[str]
    idempotence_token: str


class BotoConnector(AsyncConnector[BotoCallMetadata]):
    """Invoke one boto3 operation and return its JSON-safe response."""

    name = "Boto Connector"
    task_type_name = BOTO_TASK_TYPE
    metadata_type = BotoCallMetadata

    async def create(
        self,
        task_template: TaskTemplate,
        output_prefix: str,
        inputs: Optional[Dict[str, Any]] = None,
        task_execution_metadata: Optional[TaskExecutionMetadata] = None,
        **kwargs: Any,
    ) -> BotoCallMetadata:
        custom = MessageToDict(task_template.custom) if task_template.custom else {}
        custom = _restore_integer_values(custom)
        service = custom.get("service") or ""
        method = custom.get("method") or ""
        region = custom.get("region") or None
        if not _SERVICE_NAME.fullmatch(service):
            raise ValueError(f"Invalid boto3 service name {service!r}")
        if not _METHOD_NAME.fullmatch(method):
            raise ValueError(f"Invalid boto3 method name {method!r}")

        token = _execution_token(task_execution_metadata, custom)
        request = _replace_idempotence_token(custom.get("request") or {}, token)
        client = boto3.client(service, region_name=region, config=_BOTO_CONFIG)
        operation = getattr(client, method, None)
        if operation is None or not callable(operation):
            raise ValueError(f"Boto3 client {service!r} has no public method {method!r}")
        response = await asyncio.to_thread(operation, **request)
        response.pop("ResponseMetadata", None)

        outputs = [variable.key for variable in task_template.interface.outputs.variables]
        if len(outputs) > 1:
            raise ValueError("Boto tasks support at most one Flyte output")
        return BotoCallMetadata(
            response=_json_safe(response),
            output_key=outputs[0] if outputs else None,
            idempotence_token=token,
        )

    async def get(self, resource_meta: BotoCallMetadata, **kwargs: Any) -> Resource:
        outputs = {resource_meta.output_key: resource_meta.response} if resource_meta.output_key else {}
        return Resource(
            phase=TaskExecution.SUCCEEDED,
            outputs=outputs,
            custom_info={
                "idempotence_token": resource_meta.idempotence_token,
            },
        )

    async def delete(self, resource_meta: BotoCallMetadata, **kwargs: Any) -> None:
        return None


ConnectorRegistry.register(SageMakerTrainingConnector())
ConnectorRegistry.register(SageMakerProcessingConnector())
ConnectorRegistry.register(SageMakerTransformConnector())
ConnectorRegistry.register(BotoConnector())
