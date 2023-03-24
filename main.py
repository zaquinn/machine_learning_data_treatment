from base_census.base_census import BaseSensus
from base_credit.base_credit import BaseCredit


def main():
    BaseCredit.execute()
    BaseSensus.execute()


if __name__ == "__main__":
    main()
