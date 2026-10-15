"""A simple shared-password gate: one generic password for the whole app, like the Risk Analyzer's.

The repo is public and the app link-accessible, so this keeps casual visitors out; it is not per-user
security. Session-only: closing the tab means entering it again.

Inside the JSA admin portal the sync pre-sets ``_authed`` (``preauth`` in sync/apps.json), so the
portal's own login is the only gate there.
"""
import hmac

import streamlit as st

APP_PASSWORD = "jpsi"


def require_password() -> None:
    if st.session_state.get("_authed"):
        return

    _, center, _ = st.columns([1, 1, 1])
    with center:
        st.markdown("##### CSO & spread risk analyzer")
        st.caption("Enter the password to continue.")
        with st.form("cso_login"):          # a form, so Enter submits as well as the button
            pwd = st.text_input("Password", type="password", key="cso_password_input")
            entered = st.form_submit_button("Enter", type="primary")
        if entered:
            if hmac.compare_digest(pwd.encode(), APP_PASSWORD.encode()):
                st.session_state["_authed"] = True
                st.rerun()
            st.error("Incorrect password.")
    st.stop()
