---
id: error-lk-4044
title: Error LK-4044: Source table not found
type: error
updated: 2026-03-01
acl: public
status: current
---

# Error LK-4044: Source table not found

LK-4044 means that a table selected in the pipeline no longer exists in the source, or that the connector's user lost permission to read it.

Check that the table exists and that the connector user can read it, then refresh the source schema in the pipeline.
