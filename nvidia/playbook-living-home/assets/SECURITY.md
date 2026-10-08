# Security and privacy

Keep access tokens, household data and unredacted logs out of public issues and shared archives. Report security concerns privately to the repository maintainers using an available private reporting channel.

Living Home runs on this computer and stores household credentials and reports under the current Windows account. Use a Home Assistant account with the permissions needed for your devices and automations. Selecting devices in Living Home limits its requests; it does not change the permissions of the underlying Home Assistant token.

Keep the workspace accessible only on this computer. Remote access through a public interface, reverse proxy or tunnel is not supported. Choose **Quit** to stop it.

Back up household data privately, separately from application files and source code. Software running as your Windows user may be able to access that user's credentials.

Automations saved in Home Assistant continue independently of Living Home. Disable them there when removing access or changing their intended scope.
