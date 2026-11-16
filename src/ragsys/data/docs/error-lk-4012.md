---
id: error-lk-4012
title: Error LK-4012: Connector authentication expired
type: error
updated: 2026-03-01
acl: public
status: current
---

# Error LK-4012: Connector authentication expired

LK-4012 means that the credentials stored for a source or destination connector are no longer accepted. This usually happens after a password change or when an OAuth token is revoked.

To fix it, open Connections, select the connector and choose Reauthorize. Pipelines that use the connector resume on their next scheduled run.
