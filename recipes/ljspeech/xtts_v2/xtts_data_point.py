class DialectDataPoint:
    def __init__(
            self,
            dataset_name: str,
            sample_name: str,
            duration: float,
            speaker_id: str,
            dialect: str,
            de_text: str,
    ):
        self.dataset_name = dataset_name
        self.sample_name = sample_name
        self.duration = duration
        self.speaker_id = speaker_id
        self.dialect = dialect
        self.de_text = de_text

        split_name = self.sample_name.split("_")
        if len(split_name) > 2:
            self.orig_episode_name = '_'.join(split_name[:-1])
        else:
            self.orig_episode_name = split_name[0]

    @staticmethod
    def number_of_properties():
        return 6

    @staticmethod
    def load_single_datapoint(split_properties: list) -> DialectDataPoint:
        return DialectDataPoint(
            dataset_name=split_properties[0],
            sample_name=split_properties[1],
            duration=split_properties[2],
            speaker_id=split_properties[3],
            dialect=split_properties[4],
            de_text=split_properties[5]
        )

    def to_string(self):
        return f"{self.dataset_name}\t{self.sample_name}\t{self.duration}\t{self.speaker_id}\t{self.dialect}\t{self.de_text}\n"
