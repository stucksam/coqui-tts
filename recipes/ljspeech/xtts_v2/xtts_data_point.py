class XTTSDataPoint:
    def __init__(
            self,
            sample_name: str,
            orig_text_id: int,
            dialect: str,
            gen_de_text: str,
            gen_phoneme: str = "",
            gen_dialect: str = "",
    ):
        self.sample_name = sample_name.replace(".wav", "")
        self.orig_text_id = orig_text_id
        self.dialect = dialect
        self.gen_de_text = gen_de_text
        self.gen_phoneme = gen_phoneme
        self.gen_dialect = gen_dialect

    @staticmethod
    def number_of_properties():
        return 6

    @staticmethod
    def load_single_datapoint(split_properties: list):
        return XTTSDataPoint(
            sample_name=split_properties[0],
            orig_text_id=int(split_properties[1]),
            dialect=split_properties[2],
            gen_de_text=split_properties[3],
            gen_phoneme=split_properties[4] if len(split_properties) > 4 else "",
            gen_dialect=split_properties[5] if len(split_properties) > 5 else ""
        )

    def to_string(self) -> str:
        to_string = f"{self.sample_name}\t{self.orig_text_id}\t{self.dialect}\t{self.gen_de_text}"
        if self.gen_phoneme:
            to_string += f"\t{self.gen_phoneme}"
        if self.gen_dialect:
            to_string += f"\t{self.gen_dialect}"

        to_string += "\n"
        return to_string
