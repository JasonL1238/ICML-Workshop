"""
Principled A2J-relevant subset selection for LegalBench tasks.

Each of the 162 LegalBench tasks is classified with:
  - a2j_relevance:       high / medium / low
  - rewrite_suitability:  high / medium / low
  - auto_scoreable:       True / False
  - recommended_split:    main / appendix / exclude
  - exclusion_reason:     human-readable reason (empty string if not excluded)

Selection criteria for *main* experiment:
  - Scenario-based or legal-question-based
  - Relevant to ordinary people / Access-to-Justice settings
  - Automatically scoreable (exact-match or classification)
  - Naturally rewriteable into layperson language
  - NOT long-document contract/merger extraction (CUAD, MAUD)
  - NOT corporate/securities-heavy
  - NOT open-generation requiring manual grading
"""

from __future__ import annotations

from typing import TypedDict


class TaskMeta(TypedDict):
    task: str
    a2j_relevance: str          # high | medium | low
    rewrite_suitability: str    # high | medium | low
    auto_scoreable: bool
    recommended_split: str      # main | appendix | exclude
    exclusion_reason: str


# ── helpers ──────────────────────────────────────────────────────────────

def _entry(
    task: str,
    a2j: str,
    rw: str,
    scoreable: bool,
    split: str,
    reason: str = "",
) -> TaskMeta:
    return TaskMeta(
        task=task,
        a2j_relevance=a2j,
        rewrite_suitability=rw,
        auto_scoreable=scoreable,
        recommended_split=split,
        exclusion_reason=reason,
    )


def _cuad(task: str) -> TaskMeta:
    return _entry(task, "low", "low", True, "exclude", "CUAD long-document contract extraction")


def _maud(task: str) -> TaskMeta:
    return _entry(task, "low", "low", True, "exclude", "MAUD merger-agreement extraction")


def _supply(task: str) -> TaskMeta:
    return _entry(task, "low", "low", True, "exclude", "supply-chain corporate disclosure; not A2J-relevant")


# ── Full inventory ───────────────────────────────────────────────────────

TASK_INVENTORY: list[TaskMeta] = [
    # ─── A2J-relevant scenario/question tasks (main) ──────────────────
    _entry("abercrombie",                       "medium", "high",   True,  "main"),
    _entry("consumer_contracts_qa",             "high",   "high",   True,  "main"),
    _entry("diversity_1",                       "high",   "high",   True,  "main"),
    _entry("diversity_2",                       "high",   "high",   True,  "main"),
    _entry("diversity_3",                       "high",   "high",   True,  "main"),
    _entry("diversity_4",                       "high",   "high",   True,  "main"),
    _entry("diversity_5",                       "high",   "high",   True,  "main"),
    _entry("diversity_6",                       "high",   "high",   True,  "main"),
    _entry("hearsay",                           "high",   "high",   True,  "main"),
    _entry("insurance_policy_interpretation",   "high",   "high",   True,  "main"),
    _entry("international_citizenship_questions","medium", "high",   True,  "main"),
    _entry("learned_hands_benefits",            "high",   "high",   True,  "main"),
    _entry("learned_hands_business",            "high",   "high",   True,  "main"),
    _entry("learned_hands_consumer",            "high",   "high",   True,  "main"),
    _entry("learned_hands_courts",              "high",   "high",   True,  "main"),
    _entry("learned_hands_crime",               "high",   "high",   True,  "main"),
    _entry("learned_hands_divorce",             "high",   "high",   True,  "main"),
    _entry("learned_hands_domestic_violence",   "high",   "high",   True,  "main"),
    _entry("learned_hands_education",           "high",   "high",   True,  "main"),
    _entry("learned_hands_employment",          "high",   "high",   True,  "main"),
    _entry("learned_hands_estates",             "high",   "high",   True,  "main"),
    _entry("learned_hands_family",              "high",   "high",   True,  "main"),
    _entry("learned_hands_health",              "high",   "high",   True,  "main"),
    _entry("learned_hands_housing",             "high",   "high",   True,  "main"),
    _entry("learned_hands_immigration",         "high",   "high",   True,  "main"),
    _entry("learned_hands_torts",               "high",   "high",   True,  "main"),
    _entry("learned_hands_traffic",             "high",   "high",   True,  "main"),
    _entry("legal_reasoning_causality",         "medium", "high",   True,  "main"),
    _entry("nys_judicial_ethics",               "medium", "high",   True,  "main"),
    _entry("personal_jurisdiction",             "high",   "high",   True,  "main"),
    _entry("privacy_policy_entailment",         "high",   "high",   True,  "main"),
    _entry("privacy_policy_qa",                 "high",   "high",   True,  "main"),
    _entry("rule_qa",                           "high",   "high",   True,  "main"),
    _entry("successor_liability",               "medium", "high",   True,  "main"),
    _entry("telemarketing_sales_rule",          "high",   "high",   True,  "main"),
    _entry("ucc_v_common_law",                  "medium", "high",   True,  "main"),
    _entry("unfair_tos",                        "high",   "high",   True,  "main"),

    # ─── Appendix: scoreable but less A2J-central or harder to rewrite ─
    _entry("canada_tax_court_outcomes",         "medium", "medium", True,  "appendix"),
    _entry("contract_nli_confidentiality_of_agreement",        "medium", "medium", True, "appendix"),
    _entry("contract_nli_explicit_identification",             "medium", "medium", True, "appendix"),
    _entry("contract_nli_inclusion_of_verbally_conveyed_information", "medium", "medium", True, "appendix"),
    _entry("contract_nli_limited_use",                         "medium", "medium", True, "appendix"),
    _entry("contract_nli_no_licensing",                        "medium", "medium", True, "appendix"),
    _entry("contract_nli_notice_on_compelled_disclosure",      "medium", "medium", True, "appendix"),
    _entry("contract_nli_permissible_acquirement_of_similar_information", "medium", "medium", True, "appendix"),
    _entry("contract_nli_permissible_copy",                    "medium", "medium", True, "appendix"),
    _entry("contract_nli_permissible_development_of_similar_information", "medium", "medium", True, "appendix"),
    _entry("contract_nli_permissible_post-agreement_possession", "medium", "medium", True, "appendix"),
    _entry("contract_nli_return_of_confidential_information",  "medium", "medium", True, "appendix"),
    _entry("contract_nli_sharing_with_employees",              "medium", "medium", True, "appendix"),
    _entry("contract_nli_sharing_with_third-parties",          "medium", "medium", True, "appendix"),
    _entry("contract_nli_survival_of_obligations",             "medium", "medium", True, "appendix"),
    _entry("contract_qa",                                      "medium", "medium", True, "appendix"),
    _entry("corporate_lobbying",               "low",    "medium", True,  "appendix"),
    _entry("definition_classification",        "medium", "medium", True,  "appendix"),
    _entry("function_of_decision_section",     "medium", "low",    True,  "appendix"),
    _entry("jcrew_blocker",                    "low",    "low",    True,  "appendix"),
    _entry("oral_argument_question_purpose",   "medium", "medium", True,  "appendix"),
    _entry("overruling",                       "medium", "medium", True,  "appendix"),
    _entry("proa",                             "medium", "medium", True,  "appendix"),
    _entry("sara_entailment",                  "medium", "medium", True,  "appendix"),
    _entry("sara_numeric",                     "medium", "medium", True,  "appendix"),
    _entry("scalr",                            "medium", "medium", True,  "appendix"),
    _entry("ssla_company_defendants",          "low",    "low",    True,  "appendix"),
    _entry("ssla_individual_defendants",       "low",    "low",    True,  "appendix"),
    _entry("ssla_plaintiff",                   "low",    "low",    True,  "appendix"),
    _entry("textualism_tool_dictionaries",     "medium", "medium", True,  "appendix"),
    _entry("textualism_tool_plain",            "medium", "medium", True,  "appendix"),
    _entry("opp115_data_retention",            "medium", "medium", True,  "appendix"),
    _entry("opp115_data_security",             "medium", "medium", True,  "appendix"),
    _entry("opp115_do_not_track",              "medium", "medium", True,  "appendix"),
    _entry("opp115_first_party_collection_use","medium", "medium", True,  "appendix"),
    _entry("opp115_international_and_specific_audiences", "medium", "medium", True, "appendix"),
    _entry("opp115_policy_change",             "medium", "medium", True,  "appendix"),
    _entry("opp115_third_party_sharing_collection", "medium", "medium", True, "appendix"),
    _entry("opp115_user_access,_edit_and_deletion", "medium", "medium", True, "appendix"),
    _entry("opp115_user_choice_control",       "medium", "medium", True,  "appendix"),

    # ─── Excluded: open-generation (manual grading required) ──────────
    _entry("citation_prediction_classification","low",  "low",  True,  "exclude", "citation prediction; not A2J-relevant"),
    _entry("citation_prediction_open",          "low",  "low",  False, "exclude", "open generation; requires manual grading"),
    _entry("definition_extraction",             "low",  "low",  False, "exclude", "open extraction; requires manual grading"),

    # ─── Excluded: CUAD (long-document contract extraction) ───────────
    _cuad("cuad_affiliate_license-licensee"),
    _cuad("cuad_affiliate_license-licensor"),
    _cuad("cuad_anti-assignment"),
    _cuad("cuad_audit_rights"),
    _cuad("cuad_cap_on_liability"),
    _cuad("cuad_change_of_control"),
    _cuad("cuad_competitive_restriction_exception"),
    _cuad("cuad_covenant_not_to_sue"),
    _cuad("cuad_effective_date"),
    _cuad("cuad_exclusivity"),
    _cuad("cuad_expiration_date"),
    _cuad("cuad_governing_law"),
    _cuad("cuad_insurance"),
    _cuad("cuad_ip_ownership_assignment"),
    _cuad("cuad_irrevocable_or_perpetual_license"),
    _cuad("cuad_joint_ip_ownership"),
    _cuad("cuad_license_grant"),
    _cuad("cuad_liquidated_damages"),
    _cuad("cuad_minimum_commitment"),
    _cuad("cuad_most_favored_nation"),
    _cuad("cuad_no-solicit_of_customers"),
    _cuad("cuad_no-solicit_of_employees"),
    _cuad("cuad_non-compete"),
    _cuad("cuad_non-disparagement"),
    _cuad("cuad_non-transferable_license"),
    _cuad("cuad_notice_period_to_terminate_renewal"),
    _cuad("cuad_post-termination_services"),
    _cuad("cuad_price_restrictions"),
    _cuad("cuad_renewal_term"),
    _cuad("cuad_revenue-profit_sharing"),
    _cuad("cuad_rofr-rofo-rofn"),
    _cuad("cuad_source_code_escrow"),
    _cuad("cuad_termination_for_convenience"),
    _cuad("cuad_third_party_beneficiary"),
    _cuad("cuad_uncapped_liability"),
    _cuad("cuad_unlimited-all-you-can-eat-license"),
    _cuad("cuad_volume_restriction"),
    _cuad("cuad_warranty_duration"),

    # ─── Excluded: MAUD (merger-agreement extraction) ─────────────────
    _maud("maud_ability_to_consummate_concept_is_subject_to_mae_carveouts"),
    _maud("maud_accuracy_of_fundamental_target_rws_bringdown_standard"),
    _maud("maud_accuracy_of_target_capitalization_rw_(outstanding_shares)_bringdown_standard_answer"),
    _maud("maud_accuracy_of_target_general_rw_bringdown_timing_answer"),
    _maud("maud_additional_matching_rights_period_for_modifications_(cor)"),
    _maud("maud_application_of_buyer_consent_requirement_(negative_interim_covenant)"),
    _maud("maud_buyer_consent_requirement_(ordinary_course)"),
    _maud("maud_change_in_law__subject_to_disproportionate_impact_modifier"),
    _maud("maud_changes_in_gaap_or_other_accounting_principles__subject_to_disproportionate_impact_modifier"),
    _maud("maud_cor_permitted_in_response_to_intervening_event"),
    _maud("maud_cor_permitted_with_board_fiduciary_determination_only"),
    _maud("maud_cor_standard_(intervening_event)"),
    _maud("maud_cor_standard_(superior_offer)"),
    _maud("maud_definition_contains_knowledge_requirement_-_answer"),
    _maud("maud_definition_includes_asset_deals"),
    _maud("maud_definition_includes_stock_deals"),
    _maud("maud_fiduciary_exception__board_determination_standard"),
    _maud("maud_fiduciary_exception_board_determination_trigger_(no_shop)"),
    _maud("maud_financial_point_of_view_is_the_sole_consideration"),
    _maud("maud_fls_(mae)_standard"),
    _maud("maud_general_economic_and_financial_conditions_subject_to_disproportionate_impact_modifier"),
    _maud("maud_includes_consistent_with_past_practice"),
    _maud("maud_initial_matching_rights_period_(cor)"),
    _maud("maud_initial_matching_rights_period_(ftr)"),
    _maud("maud_intervening_event_-_required_to_occur_after_signing_-_answer"),
    _maud("maud_knowledge_definition"),
    _maud("maud_liability_standard_for_no-shop_breach_by_target_non-do_representatives"),
    _maud("maud_ordinary_course_efforts_standard"),
    _maud("maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier"),
    _maud("maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures"),
    _maud("maud_relational_language_(mae)_applies_to"),
    _maud("maud_specific_performance"),
    _maud("maud_tail_period_length"),
    _maud("maud_type_of_consideration"),

    # ─── Excluded: supply-chain corporate disclosure ──────────────────
    _supply("supply_chain_disclosure_best_practice_accountability"),
    _supply("supply_chain_disclosure_best_practice_audits"),
    _supply("supply_chain_disclosure_best_practice_certification"),
    _supply("supply_chain_disclosure_best_practice_training"),
    _supply("supply_chain_disclosure_best_practice_verification"),
    _supply("supply_chain_disclosure_disclosed_accountability"),
    _supply("supply_chain_disclosure_disclosed_audits"),
    _supply("supply_chain_disclosure_disclosed_certification"),
    _supply("supply_chain_disclosure_disclosed_training"),
    _supply("supply_chain_disclosure_disclosed_verification"),
]

# ── Lookup helpers ───────────────────────────────────────────────────────

_BY_TASK: dict[str, TaskMeta] = {t["task"]: t for t in TASK_INVENTORY}


def get_task_meta(task: str) -> TaskMeta | None:
    return _BY_TASK.get(task)


def get_tasks_for_group(group: str) -> list[str]:
    """Return sorted task names for a task-group label.

    Groups:
      main      – A2J-relevant, auto-scoreable, rewriteable subset
      appendix  – scoreable but less central tasks (robustness checks)
      all-filtered – main + appendix (everything except excluded)
    """
    if group == "main":
        return sorted(t["task"] for t in TASK_INVENTORY if t["recommended_split"] == "main")
    elif group == "appendix":
        return sorted(t["task"] for t in TASK_INVENTORY if t["recommended_split"] == "appendix")
    elif group == "all-filtered":
        return sorted(
            t["task"] for t in TASK_INVENTORY if t["recommended_split"] in ("main", "appendix")
        )
    else:
        raise ValueError(f"Unknown task group: {group!r}. Use main / appendix / all-filtered.")


def is_learned_hands(task: str) -> bool:
    """Return True if the task is a learned_hands_* task (already in layperson language)."""
    return task.startswith("learned_hands_")


def get_learned_hands_tasks() -> list[str]:
    """Return sorted list of all learned_hands_* tasks in the inventory."""
    return sorted(t["task"] for t in TASK_INVENTORY if is_learned_hands(t["task"]))


def get_legalistic_tasks(group: str = "main") -> list[str]:
    """Return sorted list of non-learned-hands tasks for a given group."""
    return sorted(
        t for t in get_tasks_for_group(group) if not is_learned_hands(t)
    )


def summary_stats() -> dict[str, int]:
    splits = {}
    for t in TASK_INVENTORY:
        s = t["recommended_split"]
        splits[s] = splits.get(s, 0) + 1
    return splits


if __name__ == "__main__":
    import csv
    import sys

    stats = summary_stats()
    print(f"Total tasks inventoried: {len(TASK_INVENTORY)}")
    for split, count in sorted(stats.items()):
        print(f"  {split:12s}: {count}")

    print(f"\nMain tasks ({stats.get('main', 0)}):")
    for t in get_tasks_for_group("main"):
        meta = get_task_meta(t)
        assert meta is not None
        print(f"  {t:45s}  a2j={meta['a2j_relevance']:6s}  rw={meta['rewrite_suitability']:6s}")

    print(f"\nAppendix tasks ({stats.get('appendix', 0)}):")
    for t in get_tasks_for_group("appendix"):
        meta = get_task_meta(t)
        assert meta is not None
        print(f"  {t:60s}  a2j={meta['a2j_relevance']:6s}  rw={meta['rewrite_suitability']:6s}")

    if "--csv" in sys.argv:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(TaskMeta.__annotations__.keys()))
        writer.writeheader()
        for t in TASK_INVENTORY:
            writer.writerow(t)
