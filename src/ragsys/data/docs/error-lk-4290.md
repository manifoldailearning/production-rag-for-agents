---
id: error-lk-4290
title: Error LK-4290: Destination rate limit exceeded
type: error
updated: 2026-03-01
acl: public
status: current
---

# Error LK-4290: Destination rate limit exceeded

LK-4290 means that the destination rejected writes because a rate limit was reached. Larkspur Cloud retries the batch automatically with exponential backoff for up to 2 hours.

If the error persists, raise the write limit in the destination, or reduce the pipeline's parallelism in its advanced settings.
