---
id: admin-guide
title: Administrator Guide
type: guide
updated: 2026-08-04
acl: public
status: current
---

# Administrator Guide

This guide covers the settings that workspace Owners and Admins manage.

## Workspaces

A workspace contains pipelines, connectors, schedules and members. Each workspace belongs to one billing account and runs in one region. An organization can have several workspaces, for example one for production and one for testing.

## Roles

There are four roles. Owners manage billing, security settings and members. Admins manage pipelines, connectors and schedules but not billing. Editors can create and edit pipelines but cannot manage connectors. Viewers can see pipelines and run history but cannot change anything.

Every workspace must have at least one Owner. An Owner cannot remove their own Owner role if they are the last Owner.

## Audit logs

The audit log records every change to pipelines, connectors, schedules, members and security settings, with the time, the member and the old and new values. Audit logs are kept for 90 days on the Business plan and for one year on the Enterprise plan. Audit logs are not available on the Starter and Team plans.

## Alerts

Pipeline failure alerts can be sent by email or to Slack. Each pipeline can have its own recipients. Alerts are sent once per failure and again when the pipeline recovers.

## Schedules

Pipelines run on a schedule or on demand. The minimum schedule interval is 60 minutes on Starter, 15 minutes on Team and 5 minutes on Business and Enterprise.

## Backfills

A backfill reloads historical data for a pipeline. A workspace can run at most two backfills at the same time. Further backfills wait in a queue.

## Maintenance

Larkspur Cloud performs scheduled maintenance on Sundays between 02:00 and 04:00 UTC. Maintenance is announced at least 72 hours in advance on the status page. Pipelines scheduled during maintenance run when it ends.
