import sys
import tabula
operations_table_area = tabula.convert_into("ddr.pdf", "ddr.json", output_format='json',guess=False,stream=True)
print(operations_table_area)

