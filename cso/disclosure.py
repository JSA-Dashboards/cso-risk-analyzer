"""The JSA disclosure shown at the top of the app.

Paragraph 2 is the firm's standard text, copied unchanged from the Risk Analyzer's disclaimer
(jsa-risk-analyzer, src/jsa_risk/ui/theme.py). Paragraph 1 is that app's model paragraph with only the
model sentence changed (Black-76 -> Bachelier spread-option model), "entered or imported" -> "entered", and
one sentence added about the example data. Compliance should confirm the adapted wording.
"""
import streamlit as st

MODEL_PARAGRAPH = (
    "All prices, Greeks, P&L, and risk figures on this dashboard are theoretical model outputs generated "
    "from the Bachelier (normal) spread-option pricing model and the position data entered by the user. "
    "They are provided for informational and analytical purposes only, do not constitute a guarantee of "
    "accuracy, and may differ materially from actual market prices, executable levels, or realized "
    "results. Verify all figures against an official broker or exchange source before making any trading "
    "or risk decision. Futures prices pulled via a live market-data feed are delayed approximately 10 "
    "minutes and are not real-time or executable quotes. Prices, volatilities, premiums and expiry dates "
    "shown in the example data are illustrative only."
)

STANDARD_PARAGRAPH = (
    "Trading commodity futures, options on futures, cash commodities, and over-the-counter derivative "
    "products involves substantial risk of loss and may not be suitable for all investors. This "
    "communication is provided for informational purposes only and does not constitute investment advice, "
    "a recommendation, or an offer or solicitation to buy or sell any futures, options, cash commodities, "
    "or derivative products. John Stewart & Associates, Inc. does not accept orders to buy or sell any "
    "financial instruments via email. The information contained herein has been obtained from sources "
    "believed to be reliable; however, its accuracy and completeness are not guaranteed. Any opinions "
    "expressed are solely those of the author, are subject to change without notice, and should not be "
    "relied upon as a basis for investment decisions. Past performance is not indicative of future "
    "results. This message may contain confidential or proprietary information intended solely for the "
    "use of the designated recipient. © John Stewart & Associates, Inc. 2026"
)


def render_disclosure() -> None:
    with st.container(border=True):
        st.caption(MODEL_PARAGRAPH)
        st.caption(STANDARD_PARAGRAPH)
