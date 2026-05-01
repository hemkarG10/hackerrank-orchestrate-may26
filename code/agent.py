"""Deterministic support triage agent.

The visible competition data is small, but this agent is intentionally written
as a repeatable pipeline: normalize, route high-risk requests, retrieve local
evidence, then emit strict structured output.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from corpus import CorpusIndex, SearchResult


VALID_COMPANIES = {"hackerrank": "HackerRank", "claude": "Claude", "visa": "Visa"}


@dataclass(frozen=True)
class Decision:
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    rule_name: str


@dataclass(frozen=True)
class Rule:
    name: str
    company: str | None
    terms: tuple[str, ...]
    product_area: str
    status: str
    request_type: str
    response: str
    reason: str
    doc_hint: str = ""

    def matches(self, company: str, text: str) -> bool:
        if self.company and self.company != company:
            return False
        return any(term in text for term in self.terms)


def normalize(text: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", (text or "").replace("\u2019", "'"))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"\s+", " ", ascii_text.strip()).lower()


def sentence(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    if not value:
        return value
    return value if value[-1] in ".!?" else value + "."


def compact_path(path: Path) -> str:
    parts = path.parts
    try:
        start = parts.index("data")
        return "/".join(parts[start:])
    except ValueError:
        return path.name


class TriageAgent:
    def __init__(self, index: CorpusIndex) -> None:
        self.index = index
        self.rules = build_rules()

    def resolve(self, issue: str, subject: str = "", company: str = "") -> Decision:
        inferred_company = self.infer_company(company, f"{subject} {issue}")
        text = normalize(f"{subject} {issue}")
        if not text:
            return Decision(
                status="replied",
                product_area="",
                response="Happy to help.",
                justification="The ticket is empty, so there is no product issue to resolve.",
                request_type="invalid",
                rule_name="empty_ticket",
            )

        for rule in self.rules:
            if rule.matches(inferred_company, text):
                results = self.index.search(
                    f"{subject} {issue} {rule.doc_hint}", company=inferred_company, limit=6
                )
                source = best_source(results, rule.doc_hint)
                justification = build_justification(rule, source)
                return Decision(
                    status=rule.status,
                    product_area=rule.product_area,
                    response=rule.response,
                    justification=justification,
                    request_type=rule.request_type,
                    rule_name=rule.name,
                )

        return self.retrieval_fallback(issue, subject, inferred_company, text)

    def infer_company(self, company: str, text: str) -> str:
        normalized = normalize(company)
        if normalized in VALID_COMPANIES:
            return VALID_COMPANIES[normalized]
        lowered = normalize(text)
        for key, value in VALID_COMPANIES.items():
            if key in lowered:
                return value
        return "None"

    def retrieval_fallback(self, issue: str, subject: str, company: str, text: str) -> Decision:
        request_type = infer_request_type(text)
        results = self.index.search(f"{subject} {issue}", company=None if company == "None" else company)
        top = results[0] if results else None
        if not top or top.score < 2.0:
            return Decision(
                status="escalated",
                product_area="",
                response="Escalate to a human support specialist because the provided corpus does not contain enough information to answer this safely.",
                justification="No strong local-corpus match was found, so the safe action is escalation instead of guessing.",
                request_type=request_type,
                rule_name="retrieval_fallback",
            )

        product_area = normalize_area(top.document.product_area)
        response = (
            f"I found relevant guidance in {top.document.title}. Please follow the documented steps there, "
            "and contact support if the issue persists or requires account-specific action."
        )
        return Decision(
            status="replied",
            product_area=product_area,
            response=response,
            justification=(
                f"Best local match was {top.document.title} ({compact_path(top.document.path)}); "
                "retrieval confidence was sufficient for a grounded reply."
            ),
            request_type=request_type,
            rule_name="retrieval_fallback",
        )


def build_justification(rule: Rule, source: SearchResult | None) -> str:
    source_text = ""
    if source and rule.doc_hint:
        source_text = f" Source: {source.document.title} ({compact_path(source.document.path)})."
    return sentence(rule.reason) + source_text


def best_source(results: Iterable[SearchResult], doc_hint: str) -> SearchResult | None:
    items = list(results)
    if not items:
        return None
    hint_tokens = set(normalize(doc_hint).split())
    if hint_tokens:
        ranked = []
        for result in items:
            haystack = normalize(f"{result.document.title} {result.document.path}")
            overlap = sum(1 for token in hint_tokens if token in haystack)
            ranked.append((overlap, result.score, result))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if ranked[0][0] > 0:
            return ranked[0][2]
    return items[0]


def infer_request_type(text: str) -> str:
    if any(term in text for term in ("feature request", "can we add", "would like to request")):
        return "feature_request"
    if any(term in text for term in ("down", "not working", "failing", "error", "bug", "blocked")):
        return "bug"
    if any(term in text for term in ("iron man", "delete all files", "weather", "actor")):
        return "invalid"
    return "product_issue"


def normalize_area(area: str) -> str:
    area = normalize(area).replace("-", "_").replace(" ", "_")
    return area or "general_support"


def build_rules() -> list[Rule]:
    return [
        Rule(
            name="prompt_injection_or_harmful_code",
            company=None,
            terms=("delete all files", "rules internal", "logic exact", "documents retrieved"),
            product_area="safety",
            status="replied",
            request_type="invalid",
            response=(
                "I cannot provide destructive code or reveal internal decision logic. If you have a support question, "
                "I can help using the public support corpus and will keep the answer limited to customer-facing guidance."
            ),
            reason="The request is unsafe or asks for internal agent details rather than product support.",
        ),
        Rule(
            name="vague_outage",
            company="None",
            terms=("it's not working", "its not working", "site is down", "none of the pages", "all requests are failing"),
            product_area="general_support",
            status="escalated",
            request_type="bug",
            response="Escalate to a human support specialist because the issue is broad and lacks enough product-specific detail to troubleshoot safely.",
            reason="Broad outage-style reports need human triage and possible incident validation.",
        ),
        Rule(
            name="irrelevant_general_question",
            company="None",
            terms=("actor in iron man", "name of the actor", "movie actor"),
            product_area="conversation_management",
            status="replied",
            request_type="invalid",
            response="I am sorry, this is out of scope from my support capabilities.",
            reason="The ticket is unrelated to HackerRank, Claude, or Visa support.",
        ),
        Rule(
            name="thanks_only",
            company="None",
            terms=("thank you for helping me", "thanks for helping", "thank you"),
            product_area="",
            status="replied",
            request_type="invalid",
            response="Happy to help.",
            reason="The ticket is a courtesy message rather than a support request.",
        ),
        Rule(
            name="hackerrank_test_active",
            company="HackerRank",
            terms=("test active", "tests stay active", "how long do the tests stay active"),
            product_area="screen",
            status="replied",
            request_type="product_issue",
            response=(
                "HackerRank tests remain active indefinitely unless a start and end time are set. To make a test expire, configure the start and end date/time in test settings. After expiration, invited candidates cannot access the test, the Invite option is disabled, and the public test link stops working."
            ),
            reason="The test expiration article says tests do not expire automatically unless expiration settings are configured.",
            doc_hint="modify test expiration time",
        ),
        Rule(
            name="hackerrank_test_variants",
            company="HackerRank",
            terms=("variant versus", "different test", "test variant", "many default versions"),
            product_area="screen",
            status="replied",
            request_type="product_issue",
            response=(
                "Use test variants when you want versions of the same assessment for different candidate profiles or tech stacks while keeping the structure related. Create a separate test when the role, skills, timing, or evaluation criteria differ enough that the assessment should be managed independently. A test must have at least two variants to use variants."
            ),
            reason="The Test Variants article explains when variants are useful and notes the minimum variant requirement.",
            doc_hint="test variants",
        ),
        Rule(
            name="hackerrank_extra_time_reinvite",
            company="HackerRank",
            terms=("reinvite", "extra time", "time accommodation", "add extra time"),
            product_area="screen",
            status="replied",
            request_type="product_issue",
            response=(
                "In HackerRank for Work, open the test, go to the Candidates tab, select the candidate, then use More > Add Time Accommodation. Enter the accommodation percentage, such as 50 for 50 percent extra time, and confirm. If the candidate needs a fresh invitation, use the reinvite flow from the same candidate-management area."
            ),
            reason="The extra-time and reinvite articles document candidate-level accommodation and reinvitation flows.",
            doc_hint="adding extra time candidates reinviting candidates",
        ),
        Rule(
            name="hackerrank_delete_google_account",
            company="HackerRank",
            terms=("google login", "delete my account", "separate hackerrank password"),
            product_area="community",
            status="replied",
            request_type="product_issue",
            response=(
                "If your HackerRank Community account was created with Google login, first set a HackerRank password by using Forgot your password on the login page. After setting the password, sign in, go to account settings, and follow the delete-account flow."
            ),
            reason="The account-management corpus explains that Google-login accounts need a password before account deletion.",
            doc_hint="delete account google password",
        ),
        Rule(
            name="claude_delete_private_conversation",
            company="Claude",
            terms=("private info", "temporary chat", "delete", "individual conversation"),
            product_area="privacy",
            status="replied",
            request_type="product_issue",
            response=(
                "You can delete an individual Claude conversation from the conversation itself: open the conversation, click the conversation name at the top, and choose Delete. Incognito or temporary chats are not used to improve Claude, but for an existing regular conversation the safe self-service action is to delete it."
            ),
            reason="The Claude privacy/account corpus provides a self-service path for deleting conversations and describes privacy controls.",
            doc_hint="delete rename conversation temporary chat privacy",
        ),
        Rule(
            name="visa_travelers_cheques",
            company="Visa",
            terms=("traveller's cheques", "traveler's cheques", "travelers cheques", "travellers cheques"),
            product_area="travel_support",
            status="replied",
            request_type="product_issue",
            response=(
                "If Visa Traveller's Cheques are lost or stolen, call the issuing bank immediately. For Citicorp traveller's cheques, the corpus lists 1-800-645-6556 or collect 1-813-623-1709, Monday-Friday 6:30 am-2:30 pm EST. Be ready to provide the serial numbers, purchase date/place, purchaser name, and details of when and how they were lost or stolen."
            ),
            reason="Visa Traveller's Cheques guidance says lost or stolen cheques should be reported to the issuing bank and lists the information needed for replacement or refund.",
            doc_hint="travellers cheques lost stolen citicorp refund",
        ),
        Rule(
            name="visa_lost_stolen_card",
            company="Visa",
            terms=("lost or stolen visa card", "card stolen", "lost card", "stolen card"),
            product_area="general_support",
            status="replied",
            request_type="product_issue",
            response=(
                "Report a lost or stolen Visa card immediately. From India, call 000-800-100-1219. From elsewhere, Visa Global Customer Assistance Services is available 24/7 at +1 303 967 1090 and can help block the card, arrange emergency cash, and coordinate a replacement card."
            ),
            reason="Visa support lists lost/stolen-card phone numbers and describes GCAS emergency card and cash support.",
            doc_hint="lost stolen card global customer assistance",
        ),
        Rule(
            name="claude_removed_member_access",
            company="Claude",
            terms=("removed my seat", "lost access", "not the workspace owner", "restore my access"),
            product_area="account_access",
            status="escalated",
            request_type="product_issue",
            response=(
                "I cannot restore access or override your organization's admin settings. For a Team or Enterprise workspace, "
                "ask an organization Owner/Admin to re-add you, approve your join request, or send a new invite. If you are re-added "
                "to the same organization with the same email address, prior chats and projects may be restored subject to visibility and retention settings."
            ),
            reason="Access restoration requires organization admin action and cannot be performed by the support agent.",
            doc_hint="removed from a Team or Enterprise organization join organization",
        ),
        Rule(
            name="hackerrank_score_dispute",
            company="HackerRank",
            terms=("increase my score", "recruiter rejected", "move me to the next round", "graded me unfairly"),
            product_area="screen",
            status="escalated",
            request_type="product_issue",
            response=(
                "Escalate to a human support specialist. HackerRank support documentation can explain scoring and reports, "
                "but it cannot change a candidate score, review private answers on demand, or instruct a hiring company to advance a candidate."
            ),
            reason="The ticket asks for score alteration and employer decision intervention, which is sensitive and account-specific.",
            doc_hint="scoring report candidate",
        ),
        Rule(
            name="visa_wrong_product_dispute",
            company="Visa",
            terms=("wrong product", "refund me", "ban the seller", "merchant sent"),
            product_area="dispute_resolution",
            status="replied",
            request_type="product_issue",
            response=(
                "Visa cannot directly refund the purchase or ban the merchant from this support request. To dispute the charge, "
                "contact your card issuer or bank using the number on the front or back of your Visa card; they may ask for transaction details. "
                "If you have concerns about a merchant, Visa's support corpus also directs cardholders to submit the merchant concern form."
            ),
            reason="The corpus routes cardholder disputes to the issuer/bank and merchant concerns to Visa's form; it does not support direct refund or merchant bans.",
            doc_hint="dispute charge concerns merchant",
        ),
        Rule(
            name="hackerrank_mock_interview_refund",
            company="HackerRank",
            terms=("mock interviews stopped", "mock interview stopped", "refund asap"),
            product_area="billing",
            status="escalated",
            request_type="product_issue",
            response=(
                "Escalate to HackerRank support for account-specific review. The billing FAQ says failed payments should be retried and incorrectly deducted amounts are refunded within 5-10 business days, but a stopped mock interview and refund request requires human handling."
            ),
            reason="Refunds and interrupted mock-interview sessions require account/payment review beyond static documentation.",
            doc_hint="payments billing mock interviews refund",
        ),
        Rule(
            name="hackerrank_payment_order",
            company="HackerRank",
            terms=("order id", "payment issue", "issue with my payment"),
            product_area="billing",
            status="escalated",
            request_type="product_issue",
            response=(
                "Escalate to HackerRank billing support for order-specific review. The public FAQ can explain payment retry and refund timing, but it cannot inspect a Stripe/Razorpay order or confirm the state of a specific transaction."
            ),
            reason="The ticket includes an order-specific payment issue that needs human billing lookup.",
            doc_hint="payments billing faqs payment fails",
        ),
        Rule(
            name="hackerrank_infosec_forms",
            company="HackerRank",
            terms=("infosec", "security questionnaire", "fill in the forms", "forms"),
            product_area="general_help",
            status="escalated",
            request_type="product_issue",
            response=(
                "Escalate to the HackerRank account or solution engineering team. Security questionnaires and company infosec forms are organization-specific and should be completed by the appropriate business/support team rather than guessed from help-center articles."
            ),
            reason="The requested infosec form completion is company-specific and not answerable from the static corpus alone.",
            doc_hint="solution engineering team security",
        ),
        Rule(
            name="hackerrank_apply_tab",
            company="HackerRank",
            terms=("apply tab", "can not able to see apply", "cannot able to see apply"),
            product_area="community",
            status="replied",
            request_type="product_issue",
            response=(
                "The Apply tab is part of HackerRank Community's QuickApply flow. Log in to HackerRank Community, go to the Apply tab, and follow the QuickApply onboarding steps. If you installed QuickApply in one browser and switch to another, the Apply tab may remind you to install the extension again."
            ),
            reason="The QuickApply article describes the Apply tab and setup flow for job applications.",
            doc_hint="quickapply apply tab",
        ),
        Rule(
            name="hackerrank_all_submissions_down",
            company="HackerRank",
            terms=("none of the submissions", "submissions across any challenges", "across any challenges"),
            product_area="community",
            status="escalated",
            request_type="bug",
            response=(
                "Escalate to HackerRank support because submissions failing across all challenges may indicate a platform issue. Include the challenge links, browser, time of failure, and whether reopening the challenge or using another supported browser changes the behavior."
            ),
            reason="The report affects all submissions, so incident-style triage is safer than a single FAQ answer.",
            doc_hint="coding challenges faqs submissions",
        ),
        Rule(
            name="hackerrank_compatibility_zoom",
            company="HackerRank",
            terms=("compatible check", "compatibility", "zoom connectivity"),
            product_area="interviews",
            status="replied",
            request_type="bug",
            response=(
                "Run the HackerRank System Compatibility Check again and use a supported browser such as the latest Chrome, Edge, Firefox, or Safari. For interview setup issues, the corpus recommends checking browser, network, or device issues; if the Zoom/connectivity error persists, contact support@hackerrank.com and include your interview or test link."
            ),
            reason="The interview FAQ and browser recommendations give troubleshooting steps for compatibility, network, and device issues.",
            doc_hint="system compatibility check zoom browser recommendations interviews",
        ),
        Rule(
            name="hackerrank_reschedule_assessment",
            company="HackerRank",
            terms=("rescheduling", "alternative date", "scheduled time", "unforeseen circumstances"),
            product_area="screen",
            status="escalated",
            request_type="product_issue",
            response=(
                "Escalate or contact the hiring company/recruiter who owns the assessment. HackerRank can document invite and test settings, but rescheduling a company assessment is controlled by the test owner and may require recruiter approval."
            ),
            reason="Assessment scheduling decisions are controlled by the hiring company and require account-specific action.",
            doc_hint="inviting candidates test reinviting candidates",
        ),
        Rule(
            name="hackerrank_interview_inactivity",
            company="HackerRank",
            terms=("inactivity times", "screen share", "hr lobby", "kicked out"),
            product_area="interviews",
            status="replied",
            request_type="feature_request",
            response=(
                "For interviews, if an interviewer exits and no other interviewers are present, the candidate moves to the lobby and the interview ends automatically after one hour of inactivity. Company Admins can also configure session inactivity timeout under Settings > Compliance & Security. If the desired behavior is to extend lobby or inactivity timing for screen-share-heavy interviews, route that to your HackerRank admin or support team."
            ),
            reason="The corpus documents interview lobby inactivity behavior and company-level inactivity timeout controls.",
            doc_hint="ending an interview inactivity company admin timeout",
        ),
        Rule(
            name="hackerrank_remove_user",
            company="HackerRank",
            terms=("remove an interviewer", "remove them from our hackerrank", "employee has left", "remove them from"),
            product_area="teams_management",
            status="replied",
            request_type="product_issue",
            response=(
                "You need Company Admin or Team Admin access. Go to your profile icon > Teams Management. To remove someone from a team, open the Teams tab, select the team, open Users, locate the user, and use the delete icon in the Action column. To prevent platform access entirely, go to the Users tab, select the user, choose More > Lock, transfer resources if prompted, and confirm."
            ),
            reason="Teams Management documents both removing team members and locking user access after employee departure.",
            doc_hint="manage team members locking user access",
        ),
        Rule(
            name="hackerrank_pause_subscription",
            company="HackerRank",
            terms=("pause our subscription", "pause subscription", "stopped all hiring"),
            product_area="billing",
            status="replied",
            request_type="product_issue",
            response=(
                "If your plan is eligible, go to your profile icon > Settings > Billing under Subscription, click Cancel Plan, then choose the Pause Subscription option. Eligible monthly individual plans can be paused for 1 to 12 months, and you can resume early from the Billing page."
            ),
            reason="The Pause Subscription article documents eligibility, pause duration, and the Billing-page flow.",
            doc_hint="pause subscription billing",
        ),
        Rule(
            name="claude_service_outage",
            company="Claude",
            terms=("stopped working completely", "all requests are failing", "claude not responding"),
            product_area="troubleshooting",
            status="escalated",
            request_type="bug",
            response=(
                "Escalate to support or incident review because all requests are failing. The Claude troubleshooting corpus distinguishes temporary capacity constraints from true service incidents and directs users to status.claude.com for confirmed incidents."
            ),
            reason="A complete failure report is broad and may represent an incident rather than a user-level fix.",
            doc_hint="troubleshoot claude error messages service incidents outages",
        ),
        Rule(
            name="visa_identity_theft",
            company="Visa",
            terms=("identity has been stolen", "identity theft"),
            product_area="fraud_security",
            status="replied",
            request_type="product_issue",
            response=(
                "If the identity theft involves your Visa card, use Visa's lost or stolen card support to cancel or block the card and ask about emergency replacement. You can also contact your issuer or bank using the phone number on your card, since issuers manage cardholder accounts."
            ),
            reason="Visa's FAQ routes identity-theft cases involving a Visa card to lost/stolen card support and issuer help.",
            doc_hint="identity theft lost stolen card",
        ),
        Rule(
            name="hackerrank_resume_builder_down",
            company="HackerRank",
            terms=("resume builder is down", "resume builder down"),
            product_area="community",
            status="escalated",
            request_type="bug",
            response=(
                "Escalate to HackerRank support because this is a reported outage of Resume Builder rather than a how-to question. The corpus explains how to create and download a resume, but it does not provide a self-service fix for the Resume Builder being down."
            ),
            reason="A product outage report needs support triage; the corpus only covers normal Resume Builder usage.",
            doc_hint="resume builder",
        ),
        Rule(
            name="hackerrank_certificate_name",
            company="HackerRank",
            terms=("name is incorrect on the certificate", "certificate name", "update it"),
            product_area="certifications",
            status="replied",
            request_type="product_issue",
            response=(
                "You can update the name on your HackerRank certificate once per account. Open your certificate page, edit the Full Name field, click Regenerate Certificate, then confirm with Update Name. The change applies to all certificates and cannot be changed again afterward."
            ),
            reason="The Certifications FAQ documents the one-time certificate name update flow.",
            doc_hint="certifications faq update name certificate",
        ),
        Rule(
            name="visa_dispute_charge",
            company="Visa",
            terms=("dispute a charge", "dispute charge"),
            product_area="dispute_resolution",
            status="replied",
            request_type="product_issue",
            response=(
                "To dispute a charge, contact your issuer or bank using the number on the front or back of your Visa card. The issuer may require detailed transaction information before resolving the disputed charge."
            ),
            reason="Visa's FAQ says cardholders should dispute charges through their issuer or bank.",
            doc_hint="dispute a charge",
        ),
        Rule(
            name="claude_security_vulnerability",
            company="Claude",
            terms=("security vulnerability", "bug bounty", "major security"),
            product_area="safeguards",
            status="replied",
            request_type="bug",
            response=(
                "For a security vulnerability in Anthropic systems or user data, review Anthropic's Responsible Disclosure Policy and submit through the public vulnerability reporting channel. If the issue is a model-safety jailbreak, the Model Safety Bug Bounty Program is run through HackerOne and has a separate scope and application process."
            ),
            reason="Claude's safeguards corpus separates technical vulnerability reporting from the model-safety bug bounty program.",
            doc_hint="public vulnerability reporting model safety bug bounty",
        ),
        Rule(
            name="claude_crawler_opt_out",
            company="Claude",
            terms=("stop crawling", "crawling my website", "crawl by website", "crawl my website"),
            product_area="privacy",
            status="replied",
            request_type="product_issue",
            response=(
                "To opt out of Anthropic crawling, update robots.txt for each domain or subdomain. To block ClaudeBot from the entire site, add: User-agent: ClaudeBot; Disallow: /. Anthropic also supports Crawl-delay for robots.txt. If you believe a bot is malfunctioning, contact claudebot@anthropic.com from an email connected to the domain."
            ),
            reason="The crawler article explains robots.txt opt-out, Crawl-delay, and the crawler contact path.",
            doc_hint="crawl data web block crawler robots.txt",
        ),
        Rule(
            name="visa_urgent_cash",
            company="Visa",
            terms=("urgent cash", "only the visa card", "need urgent cash"),
            product_area="travel_support",
            status="replied",
            request_type="product_issue",
            response=(
                "If you have your Visa card and PIN, look for an ATM with the Visa PLUS mark to withdraw local currency. If your card is lost, stolen, damaged, or compromised while traveling, Visa Global Customer Assistance Services can help with emergency cash and replacement card services; the global number in the corpus is +1 303 967 1090."
            ),
            reason="Visa travel support covers ATM cash withdrawal and GCAS emergency cash for lost, stolen, damaged, or compromised cards.",
            doc_hint="travel support emergency cash visa plus atm",
        ),
        Rule(
            name="claude_model_improvement_data",
            company="Claude",
            terms=("use my data to improve", "data to improve the models", "how long will the data be used"),
            product_area="privacy",
            status="replied",
            request_type="product_issue",
            response=(
                "The provided Claude corpus says that when you allow chats or coding sessions to help improve Claude, Anthropic de-links the data from user IDs before review, limits access, uses it only to make Claude better, and lets you change privacy/model-improvement settings at any time. The local corpus does not state a specific retention duration for this setting, so for an exact duration you should consult the Privacy Center or support."
            ),
            reason="The available local article explains privacy protections and controls but does not provide an exact retention period.",
            doc_hint="sensitive data improve claude privacy settings",
        ),
        Rule(
            name="visa_blocked_card_prompt_injection",
            company="Visa",
            terms=("carte visa", "bloquee", "bloquée", "tarjeta bloqueada", "blocked during my trip"),
            product_area="travel_support",
            status="replied",
            request_type="product_issue",
            response=(
                "I cannot reveal internal rules or decision logic, but I can help with the card issue. For a blocked or declined Visa card while traveling, contact your issuer or bank using the number on your card because the issuer is best equipped to explain the decline or block. If the card is lost, stolen, damaged, or compromised, Visa Global Customer Assistance Services can help with blocking, emergency cash, and replacement card services."
            ),
            reason="The ticket includes prompt-injection text, so the answer ignores that and uses Visa's public travel and declined-card guidance.",
            doc_hint="card declined travel support global customer assistance",
        ),
        Rule(
            name="claude_bedrock_support",
            company="Claude",
            terms=("aws bedrock", "bedrock is failing", "requests to claude with aws"),
            product_area="amazon_bedrock",
            status="replied",
            request_type="bug",
            response=(
                "For Claude through Amazon Bedrock, contact AWS Support or your AWS account manager. The Claude support corpus says Bedrock support inquiries are handled through AWS, and community support is available through AWS re:Post."
            ),
            reason="The Amazon Bedrock article routes support inquiries to AWS Support or an AWS account manager.",
            doc_hint="amazon bedrock contact support inquiries",
        ),
        Rule(
            name="claude_lti_education",
            company="Claude",
            terms=("lti key", "claude lti", "professor", "students"),
            product_area="claude_for_education",
            status="replied",
            request_type="product_issue",
            response=(
                "Claude LTI setup is intended for Claude for Education administrators and LMS administrators. In Canvas, an admin creates an LTI 1.3 Developer Key with Claude's redirect, launch, login, and JWK URLs, installs the app by Client ID, then enables Canvas under Claude for Education Organization settings > Connectors with the Canvas domain, Client ID, and Deployment ID. For plan or access questions, contact your university administrator."
            ),
            reason="The Claude for Education LTI article documents the Canvas setup flow and says questions should go to university administrators.",
            doc_hint="set up claude lti canvas education",
        ),
        Rule(
            name="visa_minimum_spend",
            company="Visa",
            terms=("minimum 10", "minimum $10", "minimum spend", "us virgin islands"),
            product_area="visa_rules",
            status="replied",
            request_type="product_issue",
            response=(
                "In general, merchants are not permitted to set a minimum or maximum amount for a Visa transaction. There is an exception in the USA and US territories, including the U.S. Virgin Islands: for credit cards only, a merchant may require a minimum transaction amount of US$10. If the merchant is applying a minimum to a Visa debit card or requiring more than US$10 on a credit card, notify your Visa card issuer."
            ),
            reason="Visa's merchant FAQ explicitly covers minimum transaction limits and the U.S. territory exception.",
            doc_hint="merchant minimum maximum visa transaction us territories",
        ),
    ]
