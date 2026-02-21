#!/usr/bin/env python3
"""
draw.io アイコン検索スクリプト

AWS / 汎用アイコンのスタイル文字列を検索して返す。
全参照ファイルを読み込む代わりにこのスクリプトで必要情報のみ抽出し、
トークン消費を最大18倍効率化する。

Usage:
    python find_drawio_icon.py "lambda"
    python find_drawio_icon.py "s3" "dynamodb" "api gateway"
    python find_drawio_icon.py --list-categories
    python find_drawio_icon.py --category compute
"""

import sys
import re
from typing import Optional

# ============================================================
# AWS Icon Database
# ============================================================

AWS_ICONS = {
    # --- Compute ---
    "ec2":          {"resIcon": "mxgraph.aws4.ec2",          "fillColor": "#ED7100", "category": "compute"},
    "lambda":       {"resIcon": "mxgraph.aws4.lambda",       "fillColor": "#ED7100", "category": "compute"},
    "ecs":          {"resIcon": "mxgraph.aws4.ecs",          "fillColor": "#ED7100", "category": "compute"},
    "eks":          {"resIcon": "mxgraph.aws4.eks",          "fillColor": "#ED7100", "category": "compute"},
    "fargate":      {"resIcon": "mxgraph.aws4.fargate",      "fillColor": "#ED7100", "category": "compute"},
    "batch":        {"resIcon": "mxgraph.aws4.batch",        "fillColor": "#ED7100", "category": "compute"},
    "lightsail":    {"resIcon": "mxgraph.aws4.lightsail",    "fillColor": "#ED7100", "category": "compute"},
    "app runner":   {"resIcon": "mxgraph.aws4.app_runner",   "fillColor": "#ED7100", "category": "compute"},
    "ecr":          {"resIcon": "mxgraph.aws4.ecr",          "fillColor": "#ED7100", "category": "compute"},

    # --- Storage ---
    "s3":           {"resIcon": "mxgraph.aws4.s3",                   "fillColor": "#3F8624", "category": "storage"},
    "ebs":          {"resIcon": "mxgraph.aws4.elastic_block_store",  "fillColor": "#3F8624", "category": "storage"},
    "efs":          {"resIcon": "mxgraph.aws4.elastic_file_system",  "fillColor": "#3F8624", "category": "storage"},
    "fsx":          {"resIcon": "mxgraph.aws4.fsx",                  "fillColor": "#3F8624", "category": "storage"},
    "glacier":      {"resIcon": "mxgraph.aws4.glacier",              "fillColor": "#3F8624", "category": "storage"},

    # --- Database ---
    "rds":          {"resIcon": "mxgraph.aws4.rds",          "fillColor": "#C925D1", "category": "database"},
    "aurora":       {"resIcon": "mxgraph.aws4.aurora",       "fillColor": "#C925D1", "category": "database"},
    "dynamodb":     {"resIcon": "mxgraph.aws4.dynamodb",     "fillColor": "#C925D1", "category": "database"},
    "elasticache":  {"resIcon": "mxgraph.aws4.elasticache",  "fillColor": "#C925D1", "category": "database"},
    "redshift":     {"resIcon": "mxgraph.aws4.redshift",     "fillColor": "#C925D1", "category": "database"},
    "neptune":      {"resIcon": "mxgraph.aws4.neptune",      "fillColor": "#C925D1", "category": "database"},
    "documentdb":   {"resIcon": "mxgraph.aws4.documentdb_with_mongodb_compatibility", "fillColor": "#C925D1", "category": "database"},
    "memorydb":     {"resIcon": "mxgraph.aws4.memorydb_for_redis", "fillColor": "#C925D1", "category": "database"},

    # --- Networking ---
    "vpc":          {"resIcon": "mxgraph.aws4.vpc",                    "fillColor": "#8C4FFF", "category": "networking"},
    "cloudfront":   {"resIcon": "mxgraph.aws4.cloudfront",            "fillColor": "#8C4FFF", "category": "networking"},
    "route 53":     {"resIcon": "mxgraph.aws4.route_53",              "fillColor": "#8C4FFF", "category": "networking"},
    "route53":      {"resIcon": "mxgraph.aws4.route_53",              "fillColor": "#8C4FFF", "category": "networking"},
    "api gateway":  {"resIcon": "mxgraph.aws4.api_gateway",           "fillColor": "#8C4FFF", "category": "networking"},
    "apigateway":   {"resIcon": "mxgraph.aws4.api_gateway",           "fillColor": "#8C4FFF", "category": "networking"},
    "elb":          {"resIcon": "mxgraph.aws4.elastic_load_balancing", "fillColor": "#8C4FFF", "category": "networking"},
    "alb":          {"resIcon": "mxgraph.aws4.elastic_load_balancing", "fillColor": "#8C4FFF", "category": "networking"},
    "nlb":          {"resIcon": "mxgraph.aws4.elastic_load_balancing", "fillColor": "#8C4FFF", "category": "networking"},
    "direct connect": {"resIcon": "mxgraph.aws4.direct_connect",      "fillColor": "#8C4FFF", "category": "networking"},
    "transit gateway": {"resIcon": "mxgraph.aws4.transit_gateway",     "fillColor": "#8C4FFF", "category": "networking"},
    "nat gateway":  {"resIcon": "mxgraph.aws4.nat_gateway",           "fillColor": "#8C4FFF", "category": "networking"},
    "privatelink":  {"resIcon": "mxgraph.aws4.privatelink",           "fillColor": "#8C4FFF", "category": "networking"},

    # --- Security ---
    "iam":              {"resIcon": "mxgraph.aws4.iam",                     "fillColor": "#DD344C", "category": "security"},
    "cognito":          {"resIcon": "mxgraph.aws4.cognito",                 "fillColor": "#DD344C", "category": "security"},
    "waf":              {"resIcon": "mxgraph.aws4.waf",                     "fillColor": "#DD344C", "category": "security"},
    "shield":           {"resIcon": "mxgraph.aws4.shield",                  "fillColor": "#DD344C", "category": "security"},
    "kms":              {"resIcon": "mxgraph.aws4.key_management_service",  "fillColor": "#DD344C", "category": "security"},
    "secrets manager":  {"resIcon": "mxgraph.aws4.secrets_manager",         "fillColor": "#DD344C", "category": "security"},
    "certificate manager": {"resIcon": "mxgraph.aws4.certificate_manager",  "fillColor": "#DD344C", "category": "security"},
    "guardduty":        {"resIcon": "mxgraph.aws4.guardduty",               "fillColor": "#DD344C", "category": "security"},
    "security hub":     {"resIcon": "mxgraph.aws4.security_hub",            "fillColor": "#DD344C", "category": "security"},

    # --- Application Integration ---
    "sqs":            {"resIcon": "mxgraph.aws4.sqs",            "fillColor": "#E7157B", "category": "integration"},
    "sns":            {"resIcon": "mxgraph.aws4.sns",            "fillColor": "#E7157B", "category": "integration"},
    "eventbridge":    {"resIcon": "mxgraph.aws4.eventbridge",    "fillColor": "#E7157B", "category": "integration"},
    "step functions": {"resIcon": "mxgraph.aws4.step_functions", "fillColor": "#E7157B", "category": "integration"},
    "appsync":        {"resIcon": "mxgraph.aws4.appsync",        "fillColor": "#E7157B", "category": "integration"},
    "mq":             {"resIcon": "mxgraph.aws4.mq",             "fillColor": "#E7157B", "category": "integration"},

    # --- Analytics ---
    "kinesis":        {"resIcon": "mxgraph.aws4.kinesis",                       "fillColor": "#8C4FFF", "category": "analytics"},
    "athena":         {"resIcon": "mxgraph.aws4.athena",                        "fillColor": "#8C4FFF", "category": "analytics"},
    "emr":            {"resIcon": "mxgraph.aws4.emr",                           "fillColor": "#8C4FFF", "category": "analytics"},
    "glue":           {"resIcon": "mxgraph.aws4.glue",                          "fillColor": "#8C4FFF", "category": "analytics"},
    "quicksight":     {"resIcon": "mxgraph.aws4.quicksight",                    "fillColor": "#8C4FFF", "category": "analytics"},
    "opensearch":     {"resIcon": "mxgraph.aws4.elasticsearch_service",         "fillColor": "#8C4FFF", "category": "analytics"},
    "elasticsearch":  {"resIcon": "mxgraph.aws4.elasticsearch_service",         "fillColor": "#8C4FFF", "category": "analytics"},
    "msk":            {"resIcon": "mxgraph.aws4.managed_streaming_for_kafka",   "fillColor": "#8C4FFF", "category": "analytics"},
    "lake formation": {"resIcon": "mxgraph.aws4.lake_formation",                "fillColor": "#8C4FFF", "category": "analytics"},

    # --- Management ---
    "cloudwatch":     {"resIcon": "mxgraph.aws4.cloudwatch",       "fillColor": "#E7157B", "category": "management"},
    "cloudformation": {"resIcon": "mxgraph.aws4.cloudformation",   "fillColor": "#E7157B", "category": "management"},
    "cloudtrail":     {"resIcon": "mxgraph.aws4.cloudtrail",       "fillColor": "#E7157B", "category": "management"},
    "systems manager": {"resIcon": "mxgraph.aws4.systems_manager", "fillColor": "#E7157B", "category": "management"},
    "config":         {"resIcon": "mxgraph.aws4.config",           "fillColor": "#E7157B", "category": "management"},
    "x-ray":          {"resIcon": "mxgraph.aws4.xray",             "fillColor": "#E7157B", "category": "management"},

    # --- AI/ML ---
    "sagemaker":    {"resIcon": "mxgraph.aws4.sagemaker",    "fillColor": "#01A88D", "category": "ai_ml"},
    "bedrock":      {"resIcon": "mxgraph.aws4.bedrock",      "fillColor": "#01A88D", "category": "ai_ml"},
    "comprehend":   {"resIcon": "mxgraph.aws4.comprehend",   "fillColor": "#01A88D", "category": "ai_ml"},
    "rekognition":  {"resIcon": "mxgraph.aws4.rekognition",  "fillColor": "#01A88D", "category": "ai_ml"},
    "lex":          {"resIcon": "mxgraph.aws4.lex",          "fillColor": "#01A88D", "category": "ai_ml"},
    "polly":        {"resIcon": "mxgraph.aws4.polly",        "fillColor": "#01A88D", "category": "ai_ml"},
    "translate":    {"resIcon": "mxgraph.aws4.translate",     "fillColor": "#01A88D", "category": "ai_ml"},
    "textract":     {"resIcon": "mxgraph.aws4.textract",     "fillColor": "#01A88D", "category": "ai_ml"},

    # --- Developer Tools ---
    "codecommit":   {"resIcon": "mxgraph.aws4.codecommit",   "fillColor": "#C925D1", "category": "devtools"},
    "codebuild":    {"resIcon": "mxgraph.aws4.codebuild",    "fillColor": "#C925D1", "category": "devtools"},
    "codedeploy":   {"resIcon": "mxgraph.aws4.codedeploy",   "fillColor": "#C925D1", "category": "devtools"},
    "codepipeline": {"resIcon": "mxgraph.aws4.codepipeline", "fillColor": "#C925D1", "category": "devtools"},
}

# AWS Group styles
AWS_GROUPS = {
    "aws cloud":        {"grIcon": "mxgraph.aws4.group_aws_cloud",          "strokeColor": "#242F3E", "dashed": "0"},
    "region":           {"grIcon": "mxgraph.aws4.group_region",             "strokeColor": "#00A4A6", "dashed": "1"},
    "vpc":              {"grIcon": "mxgraph.aws4.group_vpc2",               "strokeColor": "#8C4FFF", "dashed": "0"},
    "availability zone": {"grIcon": "mxgraph.aws4.group_availability_zone", "strokeColor": "#00A4A6", "dashed": "1"},
    "az":               {"grIcon": "mxgraph.aws4.group_availability_zone",  "strokeColor": "#00A4A6", "dashed": "1"},
    "public subnet":    {"grIcon": "mxgraph.aws4.group_security_group",     "strokeColor": "#7AA116", "dashed": "0"},
    "private subnet":   {"grIcon": "mxgraph.aws4.group_security_group",     "strokeColor": "#147EBA", "dashed": "0"},
    "security group":   {"grIcon": "mxgraph.aws4.group_security_group",     "strokeColor": "#DD344C", "dashed": "1"},
    "auto scaling":     {"grIcon": "mxgraph.aws4.group_auto_scaling_group", "strokeColor": "#ED7100", "dashed": "1"},
    "account":          {"grIcon": "mxgraph.aws4.group_account",            "strokeColor": "#E7157B", "dashed": "0"},
}

# External actors
AWS_ACTORS = {
    "user":     "shape=mxgraph.aws4.user;outlineConnect=0;fontColor=#232F3E;sketch=0;",
    "client":   "shape=mxgraph.aws4.client;outlineConnect=0;fontColor=#232F3E;sketch=0;",
    "internet": "shape=mxgraph.aws4.internet_alt2;outlineConnect=0;fontColor=#232F3E;sketch=0;",
    "server":   "shape=mxgraph.aws4.traditional_server;outlineConnect=0;fontColor=#232F3E;sketch=0;",
    "datacenter": "shape=mxgraph.aws4.traditional_server;outlineConnect=0;fontColor=#232F3E;sketch=0;",
}


def build_resource_style(icon_data: dict, label: str = "") -> str:
    """Build complete mxCell style string for an AWS resource icon."""
    return (
        f"sketch=0;outlineConnect=0;fontColor=#232F3E;"
        f"gradientColor=none;fillColor={icon_data['fillColor']};strokeColor=#ffffff;"
        f"fontFamily=Helvetica;fontSize=12;"
        f"verticalLabelPosition=bottom;verticalAlign=top;align=center;"
        f"html=1;aspect=fixed;"
        f"shape=mxgraph.aws4.resourceIcon;resIcon={icon_data['resIcon']};"
    )


def build_group_style(group_data: dict) -> str:
    """Build complete mxCell style string for an AWS group."""
    dashed_str = f"dashed={group_data['dashed']};" if group_data['dashed'] == "1" else ""
    return (
        f"points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],"
        f"[1,0.25],[1,0.5],[1,0.75],[1,1],"
        f"[0.75,1],[0.5,1],[0.25,1],[0,1],"
        f"[0,0.75],[0,0.5],[0,0.25]];"
        f"outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;"
        f"fontSize=14;fontStyle=1;fontFamily=Helvetica;fontColor=#232F3E;"
        f"container=1;pointerEvents=0;collapsible=0;recursiveResize=0;"
        f"shape=mxgraph.aws4.group;grIcon={group_data['grIcon']};"
        f"strokeColor={group_data['strokeColor']};fillColor=none;"
        f"{dashed_str}"
        f"verticalAlign=top;align=left;spacingLeft=30;"
    )


def search(query: str) -> list[dict]:
    """Search for icons matching the query."""
    query_lower = query.lower().strip()
    results = []

    # Exact match first
    if query_lower in AWS_ICONS:
        icon = AWS_ICONS[query_lower]
        results.append({
            "name": query_lower,
            "type": "resource",
            "style": build_resource_style(icon),
            "category": icon["category"],
        })

    if query_lower in AWS_GROUPS:
        group = AWS_GROUPS[query_lower]
        results.append({
            "name": query_lower,
            "type": "group",
            "style": build_group_style(group),
        })

    if query_lower in AWS_ACTORS:
        results.append({
            "name": query_lower,
            "type": "actor",
            "style": AWS_ACTORS[query_lower],
        })

    # Fuzzy match (substring)
    if not results:
        for name, icon in AWS_ICONS.items():
            if query_lower in name or query_lower in icon["resIcon"].lower():
                results.append({
                    "name": name,
                    "type": "resource",
                    "style": build_resource_style(icon),
                    "category": icon["category"],
                })
        for name, group in AWS_GROUPS.items():
            if query_lower in name:
                results.append({
                    "name": name,
                    "type": "group",
                    "style": build_group_style(group),
                })
        for name, style in AWS_ACTORS.items():
            if query_lower in name:
                results.append({
                    "name": name,
                    "type": "actor",
                    "style": style,
                })

    return results


def list_categories() -> dict[str, list[str]]:
    """List all categories and their services."""
    cats: dict[str, list[str]] = {}
    for name, icon in AWS_ICONS.items():
        cat = icon["category"]
        if cat not in cats:
            cats[cat] = []
        cats[cat].append(name)
    return cats


def main():
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python find_drawio_icon.py <query> [query2] ...")
        print("  python find_drawio_icon.py --list-categories")
        print("  python find_drawio_icon.py --category <category_name>")
        print()
        print("Examples:")
        print('  python find_drawio_icon.py "lambda"')
        print('  python find_drawio_icon.py "s3" "dynamodb"')
        print('  python find_drawio_icon.py --category compute')
        sys.exit(0)

    if args[0] == "--list-categories":
        cats = list_categories()
        for cat, services in sorted(cats.items()):
            print(f"\n[{cat}] ({len(services)} services)")
            for s in sorted(services):
                print(f"  - {s}")
        return

    if args[0] == "--category" and len(args) > 1:
        cat_name = args[1].lower()
        cats = list_categories()
        if cat_name in cats:
            print(f"\n[{cat_name}] ({len(cats[cat_name])} services)")
            for s in sorted(cats[cat_name]):
                icon = AWS_ICONS[s]
                print(f"\n  {s}:")
                print(f"    style=\"{build_resource_style(icon)}\"")
        else:
            print(f"Category '{cat_name}' not found.")
            print(f"Available: {', '.join(sorted(cats.keys()))}")
        return

    # Search for each query
    for query in args:
        results = search(query)
        if results:
            print(f"\n=== Results for '{query}' ({len(results)} found) ===")
            for r in results:
                print(f"\n  [{r['type'].upper()}] {r['name']}")
                if r['type'] == 'resource':
                    print(f"  category: {r['category']}")
                print(f"  style=\"{r['style']}\"")
        else:
            print(f"\n=== No results for '{query}' ===")
            # Suggest similar
            query_lower = query.lower()
            suggestions = [
                name for name in AWS_ICONS
                if any(part in name for part in query_lower.split())
            ]
            if suggestions:
                print(f"  Did you mean: {', '.join(suggestions[:5])}?")


if __name__ == "__main__":
    main()
