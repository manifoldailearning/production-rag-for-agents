---
id: error-lk-5100
title: Error LK-5100: Worker out of memory
type: error
updated: 2026-03-01
acl: public
status: current
---

# Error LK-5100: Worker out of memory

LK-5100 means that a sync worker ran out of memory, usually while processing very wide rows or large JSON columns.

Reduce the batch size in the pipeline's advanced settings, or choose a larger worker size. Larger workers are available on Business and Enterprise plans.
