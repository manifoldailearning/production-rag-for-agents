---
id: data-retention
title: Data Retention and Deletion
type: policy
updated: 2026-03-01
acl: public
status: current
---

# Data Retention and Deletion

Larkspur Cloud does not store the rows that pipelines move. Rows are streamed from the source to the destination and are held in encrypted buffers for at most 24 hours while a sync is in progress.

Pipeline run logs are kept for the log retention period of your plan, listed on the pricing page.

When a subscription is cancelled, the workspace is retained for 30 days so that you can export its configuration. After 30 days, the workspace, its pipeline configuration and its logs are permanently deleted. Deletion cannot be undone.

You can request immediate deletion of a workspace by opening a ticket from an Owner account.
