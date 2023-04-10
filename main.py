from base_census.base_census import BaseCensus
from base_credit.base_credit import BaseCredit
from base_risk_credit.base_risk_credit import BaseRiskCredit


def main():
    # BaseCredit.data_pre_processment()
    # BaseCensus.data_pre_processment()
    # BaseRiskCredit.data_pre_processment()
    BaseCensus.execute_decision_tree()
    # BaseCredit.execute_naive_bayes()
    # BaseCensus.execute_naive_bayes()


if __name__ == "__main__":
    main()
