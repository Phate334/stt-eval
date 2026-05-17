from dataclasses import dataclass


@dataclass(frozen=True)
class CerResult:
    substitutions: int
    insertions: int
    deletions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def ratio(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.reference_length


@dataclass(frozen=True)
class _Cell:
    cost: int
    substitutions: int
    insertions: int
    deletions: int


def calculate_cer(reference: str, hypothesis: str) -> CerResult:
    """Return CER edit counts from normalized reference and hypothesis."""

    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    table: list[list[_Cell]] = []

    for row in range(rows):
        table_row = []
        for column in range(columns):
            if row == 0:
                table_row.append(_Cell(column, 0, column, 0))
            elif column == 0:
                table_row.append(_Cell(row, 0, 0, row))
            else:
                table_row.append(_Cell(0, 0, 0, 0))
        table.append(table_row)

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                table[row][column] = table[row - 1][column - 1]
                continue

            substitution = table[row - 1][column - 1]
            insertion = table[row][column - 1]
            deletion = table[row - 1][column]
            candidates = (
                _Cell(
                    substitution.cost + 1,
                    substitution.substitutions + 1,
                    substitution.insertions,
                    substitution.deletions,
                ),
                _Cell(
                    insertion.cost + 1,
                    insertion.substitutions,
                    insertion.insertions + 1,
                    insertion.deletions,
                ),
                _Cell(
                    deletion.cost + 1,
                    deletion.substitutions,
                    deletion.insertions,
                    deletion.deletions + 1,
                ),
            )
            table[row][column] = min(
                candidates,
                key=lambda cell: (
                    cell.cost,
                    cell.substitutions,
                    cell.deletions,
                    cell.insertions,
                ),
            )

    final = table[-1][-1]
    return CerResult(
        substitutions=final.substitutions,
        insertions=final.insertions,
        deletions=final.deletions,
        reference_length=len(reference),
    )
