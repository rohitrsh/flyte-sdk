from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from flyteidl2.core.execution_pb2 import TaskExecution
from flyteidl2.core.interface_pb2 import TypedInterface, Variable, VariableEntry, VariableMap
from flyteidl2.core.tasks_pb2 import TaskTemplate
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from google.protobuf.json_format import ParseDict
from google.protobuf.struct_pb2 import Struct

from flyteplugins.sagemaker import connector as connector_module
from flyteplugins.sagemaker.connector import BotoConnector
from flyteplugins.sagemaker.task import Boto


def task_template(
    *,
    service: str = "sagemaker",
    method: str = "create_model",
) -> TaskTemplate:
    custom = Struct()
    ParseDict(
        {
            "service": service,
            "method": method,
            "region": "us-west-2",
            "request": {
                "ModelName": "model-{idempotence_token}",
                "Nested": {
                    "Count": 1,
                    "Token": "{idempotence_token}",
                },
            },
        },
        custom,
    )
    output_type = LiteralType(simple=SimpleType.STRUCT)
    interface = TypedInterface(
        outputs=VariableMap(
            variables=[
                VariableEntry(
                    key="result",
                    value=Variable(type=output_type),
                )
            ]
        )
    )
    return TaskTemplate(type="boto", custom=custom, interface=interface)


@pytest.mark.asyncio
async def test_boto_connector_replaces_token_and_returns_json_safe_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operation = MagicMock(
        return_value={
            "ModelArn": "arn:aws:sagemaker:us-west-2:123456789012:model/example",
            "CreatedAt": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "Cost": Decimal(10),
            "ResponseMetadata": {"RequestId": "request"},
        }
    )
    client = MagicMock()
    client.create_model = operation
    monkeypatch.setattr(connector_module.boto3, "client", lambda *args, **kwargs: client)
    connector = BotoConnector()

    metadata = await connector.create(
        task_template(),
        output_prefix="s3://outputs",
    )
    result = await connector.get(metadata)

    request = operation.call_args.kwargs
    assert request["ModelName"].startswith("model-")
    assert request["Nested"]["Count"] == 1
    assert request["Nested"]["Token"] == metadata.idempotence_token
    assert result.phase == TaskExecution.SUCCEEDED
    assert result.outputs["result"] == {
        "ModelArn": "arn:aws:sagemaker:us-west-2:123456789012:model/example",
        "CreatedAt": "2026-01-01T00:00:00+00:00",
        "Cost": 10,
    }


@pytest.mark.parametrize(
    ("service", "method"),
    [
        ("", "create_model"),
        ("sagemaker", "_private"),
        ("SageMaker", "create_model"),
        ("sagemaker", "CreateModel"),
    ],
)
@pytest.mark.asyncio
async def test_boto_connector_rejects_invalid_operation_names(
    service: str,
    method: str,
) -> None:
    with pytest.raises(ValueError):
        await BotoConnector().create(
            task_template(service=service, method=method),
            output_prefix="s3://outputs",
        )


def test_boto_config_serializes_exact_request() -> None:
    config = Boto(
        service="s3",
        method="put_object",
        region="us-west-2",
        request={
            "Bucket": "example-bucket",
            "Key": "results/{idempotence_token}",
        },
    )

    assert config.to_custom_config() == {
        "service": "s3",
        "method": "put_object",
        "region": "us-west-2",
        "request": {
            "Bucket": "example-bucket",
            "Key": "results/{idempotence_token}",
        },
    }
