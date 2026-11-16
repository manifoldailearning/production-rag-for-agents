---
id: postmortem-2026-07
title: Postmortem: eu-west Outage on 14 July 2026
type: postmortem
updated: 2026-07-18
acl: public
status: current
---

# Postmortem: eu-west Outage on 14 July 2026

On 14 July 2026, pipelines in the eu-west region failed for 47 minutes between 09:12 and 09:59 UTC.

The cause was an expired TLS certificate on an internal load balancer. Certificate renewal had been automated, but the renewal job had been failing silently for two weeks.

No customer data was lost. Failed syncs were retried automatically after recovery. Business and Enterprise customers in eu-west received the 10 percent service credit automatically, without needing to request it.

Follow-up actions: alert on certificate expiry 21 days in advance, and alert when the renewal job fails.
