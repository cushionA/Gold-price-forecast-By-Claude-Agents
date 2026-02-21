# AWS Icons for draw.io

## 使い方

このファイルを丸ごと読み込まず、`find_drawio_icon.py` を使って必要なアイコンのみ取得すること。

```bash
python .claude/skills/drawio/scripts/find_drawio_icon.py "lambda"
python .claude/skills/drawio/scripts/find_drawio_icon.py "ec2"
python .claude/skills/drawio/scripts/find_drawio_icon.py "s3"
```

---

## AWS アイコンスタイルの基本形

### リソースアイコン（個別サービス）

```
shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.{service};
aspect=fixed;whiteSpace=wrap;html=1;
fontFamily=Helvetica;fontSize=12;
verticalLabelPosition=bottom;verticalAlign=top;align=center;
```

推奨サイズ: `width="78" height="78"`

### グループアイコン（VPC, Region, AZ など）

```
points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];
outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;
fontSize=12;fontStyle=1;fontFamily=Helvetica;
container=1;pointerEvents=0;collapsible=0;recursiveResize=0;
shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_{group_type};
strokeColor={color};fillColor=none;verticalAlign=top;align=left;spacingLeft=30;
```

---

## Compute

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| EC2 | `mxgraph.aws4.ec2` | `#ED7100` | `#ffffff` |
| Lambda | `mxgraph.aws4.lambda` | `#ED7100` | `#ffffff` |
| ECS | `mxgraph.aws4.ecs` | `#ED7100` | `#ffffff` |
| EKS | `mxgraph.aws4.eks` | `#ED7100` | `#ffffff` |
| Fargate | `mxgraph.aws4.fargate` | `#ED7100` | `#ffffff` |
| Batch | `mxgraph.aws4.batch` | `#ED7100` | `#ffffff` |
| Lightsail | `mxgraph.aws4.lightsail` | `#ED7100` | `#ffffff` |
| App Runner | `mxgraph.aws4.app_runner` | `#ED7100` | `#ffffff` |

## Storage

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| S3 | `mxgraph.aws4.s3` | `#3F8624` | `#ffffff` |
| EBS | `mxgraph.aws4.elastic_block_store` | `#3F8624` | `#ffffff` |
| EFS | `mxgraph.aws4.elastic_file_system` | `#3F8624` | `#ffffff` |
| FSx | `mxgraph.aws4.fsx` | `#3F8624` | `#ffffff` |
| Glacier | `mxgraph.aws4.glacier` | `#3F8624` | `#ffffff` |

## Database

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| RDS | `mxgraph.aws4.rds` | `#C925D1` | `#ffffff` |
| Aurora | `mxgraph.aws4.aurora` | `#C925D1` | `#ffffff` |
| DynamoDB | `mxgraph.aws4.dynamodb` | `#C925D1` | `#ffffff` |
| ElastiCache | `mxgraph.aws4.elasticache` | `#C925D1` | `#ffffff` |
| Redshift | `mxgraph.aws4.redshift` | `#C925D1` | `#ffffff` |
| Neptune | `mxgraph.aws4.neptune` | `#C925D1` | `#ffffff` |
| DocumentDB | `mxgraph.aws4.documentdb_with_mongodb_compatibility` | `#C925D1` | `#ffffff` |
| MemoryDB | `mxgraph.aws4.memorydb_for_redis` | `#C925D1` | `#ffffff` |

## Networking

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| VPC | `mxgraph.aws4.vpc` | `#8C4FFF` | `#ffffff` |
| CloudFront | `mxgraph.aws4.cloudfront` | `#8C4FFF` | `#ffffff` |
| Route 53 | `mxgraph.aws4.route_53` | `#8C4FFF` | `#ffffff` |
| API Gateway | `mxgraph.aws4.api_gateway` | `#8C4FFF` | `#ffffff` |
| ELB/ALB | `mxgraph.aws4.elastic_load_balancing` | `#8C4FFF` | `#ffffff` |
| Direct Connect | `mxgraph.aws4.direct_connect` | `#8C4FFF` | `#ffffff` |
| Transit Gateway | `mxgraph.aws4.transit_gateway` | `#8C4FFF` | `#ffffff` |
| NAT Gateway | `mxgraph.aws4.nat_gateway` | `#8C4FFF` | `#ffffff` |
| PrivateLink | `mxgraph.aws4.privatelink` | `#8C4FFF` | `#ffffff` |

## Security

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| IAM | `mxgraph.aws4.iam` | `#DD344C` | `#ffffff` |
| Cognito | `mxgraph.aws4.cognito` | `#DD344C` | `#ffffff` |
| WAF | `mxgraph.aws4.waf` | `#DD344C` | `#ffffff` |
| Shield | `mxgraph.aws4.shield` | `#DD344C` | `#ffffff` |
| KMS | `mxgraph.aws4.key_management_service` | `#DD344C` | `#ffffff` |
| Secrets Manager | `mxgraph.aws4.secrets_manager` | `#DD344C` | `#ffffff` |
| Certificate Manager | `mxgraph.aws4.certificate_manager` | `#DD344C` | `#ffffff` |
| GuardDuty | `mxgraph.aws4.guardduty` | `#DD344C` | `#ffffff` |
| Security Hub | `mxgraph.aws4.security_hub` | `#DD344C` | `#ffffff` |

## Application Integration

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| SQS | `mxgraph.aws4.sqs` | `#E7157B` | `#ffffff` |
| SNS | `mxgraph.aws4.sns` | `#E7157B` | `#ffffff` |
| EventBridge | `mxgraph.aws4.eventbridge` | `#E7157B` | `#ffffff` |
| Step Functions | `mxgraph.aws4.step_functions` | `#E7157B` | `#ffffff` |
| AppSync | `mxgraph.aws4.appsync` | `#E7157B` | `#ffffff` |
| MQ | `mxgraph.aws4.mq` | `#E7157B` | `#ffffff` |

## Analytics

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| Kinesis | `mxgraph.aws4.kinesis` | `#8C4FFF` | `#ffffff` |
| Athena | `mxgraph.aws4.athena` | `#8C4FFF` | `#ffffff` |
| EMR | `mxgraph.aws4.emr` | `#8C4FFF` | `#ffffff` |
| Glue | `mxgraph.aws4.glue` | `#8C4FFF` | `#ffffff` |
| QuickSight | `mxgraph.aws4.quicksight` | `#8C4FFF` | `#ffffff` |
| OpenSearch | `mxgraph.aws4.elasticsearch_service` | `#8C4FFF` | `#ffffff` |
| MSK | `mxgraph.aws4.managed_streaming_for_kafka` | `#8C4FFF` | `#ffffff` |
| Lake Formation | `mxgraph.aws4.lake_formation` | `#8C4FFF` | `#ffffff` |

## Management & Monitoring

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| CloudWatch | `mxgraph.aws4.cloudwatch` | `#E7157B` | `#ffffff` |
| CloudFormation | `mxgraph.aws4.cloudformation` | `#E7157B` | `#ffffff` |
| CloudTrail | `mxgraph.aws4.cloudtrail` | `#E7157B` | `#ffffff` |
| Systems Manager | `mxgraph.aws4.systems_manager` | `#E7157B` | `#ffffff` |
| Config | `mxgraph.aws4.config` | `#E7157B` | `#ffffff` |
| X-Ray | `mxgraph.aws4.xray` | `#E7157B` | `#ffffff` |

## AI/ML

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| SageMaker | `mxgraph.aws4.sagemaker` | `#01A88D` | `#ffffff` |
| Bedrock | `mxgraph.aws4.bedrock` | `#01A88D` | `#ffffff` |
| Comprehend | `mxgraph.aws4.comprehend` | `#01A88D` | `#ffffff` |
| Rekognition | `mxgraph.aws4.rekognition` | `#01A88D` | `#ffffff` |
| Lex | `mxgraph.aws4.lex` | `#01A88D` | `#ffffff` |
| Polly | `mxgraph.aws4.polly` | `#01A88D` | `#ffffff` |
| Translate | `mxgraph.aws4.translate` | `#01A88D` | `#ffffff` |
| Textract | `mxgraph.aws4.textract` | `#01A88D` | `#ffffff` |

## Developer Tools

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| CodeCommit | `mxgraph.aws4.codecommit` | `#C925D1` | `#ffffff` |
| CodeBuild | `mxgraph.aws4.codebuild` | `#C925D1` | `#ffffff` |
| CodeDeploy | `mxgraph.aws4.codedeploy` | `#C925D1` | `#ffffff` |
| CodePipeline | `mxgraph.aws4.codepipeline` | `#C925D1` | `#ffffff` |

## Containers

| Service | resIcon | fillColor | strokeColor |
|---------|---------|-----------|-------------|
| ECR | `mxgraph.aws4.ecr` | `#ED7100` | `#ffffff` |

## Groups (for nesting)

| Group | grIcon | strokeColor | dashed |
|-------|--------|-------------|--------|
| AWS Cloud | `mxgraph.aws4.group_aws_cloud` | `#242F3E` | 0 |
| Region | `mxgraph.aws4.group_region` | `#00A4A6` | 1 |
| VPC | `mxgraph.aws4.group_vpc2` | `#8C4FFF` | 0 |
| Availability Zone | `mxgraph.aws4.group_availability_zone` | `#00A4A6` | 1 |
| Public Subnet | `mxgraph.aws4.group_security_group` | `#7AA116` | 0 |
| Private Subnet | `mxgraph.aws4.group_security_group` | `#147EBA` | 0 |
| Security Group | `mxgraph.aws4.group_security_group` | `#DD344C` | 1 |
| Auto Scaling | `mxgraph.aws4.group_auto_scaling_group` | `#ED7100` | 1 |
| Account | `mxgraph.aws4.group_account` | `#E7157B` | 0 |

## 外部アクター

| Actor | style |
|-------|-------|
| ユーザー | `shape=mxgraph.aws4.user;outlineConnect=0;fontColor=#232F3E;sketch=0;` |
| Client | `shape=mxgraph.aws4.client;outlineConnect=0;fontColor=#232F3E;sketch=0;` |
| Internet | `shape=mxgraph.aws4.internet_alt2;outlineConnect=0;fontColor=#232F3E;sketch=0;` |
| Corporate Data Center | `shape=mxgraph.aws4.traditional_server;outlineConnect=0;fontColor=#232F3E;sketch=0;` |

## 注意事項

- **OpenSearch** は `elasticsearch_service` で登録されている（旧名称）
- **ALB/NLB/CLB** はすべて `elastic_load_balancing` を使用
- アイコンが見つからない場合は汎用の `mxgraph.aws4.resourceIcon` + 適切な `fillColor` で代用
- `aspect=fixed;` を付けないとアイコンが歪む
