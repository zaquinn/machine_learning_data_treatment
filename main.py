from base_census.base_census import BaseSensus
from base_credit.base_credit import BaseCredit
from base_risk_credit.base_risk_credit import BaseRiskCredit


def main():
    # BaseCredit.data_pre_processment()
    # BaseSensus.data_pre_processment()
    # BaseRiskCredit.data_pre_processment()
    BaseCredit.execute_decision_tree()
    # BaseCredit.execute_naive_bayes()
    # BaseSensus.execute_naive_bayes()


if __name__ == "__main__":
    main()
