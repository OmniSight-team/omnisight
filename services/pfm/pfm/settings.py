from omnisight_common import CommonSettings


class Settings(CommonSettings):
    app_name: str = "omnisight-pfm"

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # The rule detector (OMNI-908, D-52).
    # How often it looks at the cameras' newest frames.
    rules_interval_s: float = 2.0
    # How far back each look reads; a run that reaches further back is followed back to its start.
    rules_lookback_s: float = 120.0
    # The longest pause between two motion frames that still counts as one run. A live camera
    # stores its motion frames at its own frame rate, and a heartbeat every second (D-52).
    rules_max_gap_s: float = 2.0
    # "Activity outside hours": the run ends after this long without motion (D-52).
    rules_quiet_s: float = 60.0
    # How far back a run is followed at most.
    rules_max_history_s: float = 7200.0
    # The time zone of the cameras' watched hours.
    rules_timezone: str = "Asia/Jerusalem"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
