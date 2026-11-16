---
id: error-lk-5003
title: Error LK-5003: Schema drift detected
type: error
updated: 2026-08-04
acl: public
status: current
---

# Error LK-5003: Schema drift detected

LK-5003 means that the source schema changed in a way that does not match the destination table, for example a new column or a changed type.

From release 4.5, pipelines with automatic schema evolution turned on add new columns to the destination and resolve LK-5003 without stopping. Type changes still stop the pipeline.

To resolve it manually, open the pipeline, review the detected changes and choose Apply, or update the column mapping.
