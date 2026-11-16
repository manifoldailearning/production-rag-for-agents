---
id: data-residency
title: Regions and Data Residency
type: policy
updated: 2026-10-20
acl: public
status: current
---

# Regions and Data Residency

Each workspace runs in one region, chosen when the workspace is created. The available regions are us-east, eu-west and ap-south. The ap-south region was added in release 4.6.

Workers, buffers and logs for a workspace stay in its region. Workspaces in eu-west never process data outside the European Union.

A workspace cannot be moved to another region. To change region, create a new workspace in the target region and recreate the pipelines.
