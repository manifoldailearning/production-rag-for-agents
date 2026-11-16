---
id: sso-setup
title: Setting Up Single Sign-On
type: howto
updated: 2026-06-02
acl: public
status: current
---

# Setting Up Single Sign-On

Larkspur Cloud supports SAML 2.0 single sign-on with any compliant identity provider.

SSO is available on Business and Enterprise plans, and on Team with the SSO add-on.

To set it up, a workspace Owner opens Settings, then Security, then Single sign-on, and uploads the identity provider's metadata file. Larkspur Cloud shows the assertion consumer service URL and entity ID to enter in the identity provider.

After testing the connection, the Owner can turn on Enforce SSO. When SSO is enforced, members can no longer sign in with a password. Owners keep a password login as a recovery path.
