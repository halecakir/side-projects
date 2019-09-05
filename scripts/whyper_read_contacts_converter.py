"""TODO"""
import xlrd
import xlwt


def modify_whyper_data(file_path, out_file):
    """TODO"""
    loc = (file_path)
    workbook = xlrd.open_workbook(loc)
    sheet = workbook.sheet_by_index(0)
    file_name = file_path.split("/")[-1]
    if file_name == "Read_Contacts.xls":
        data = []
        for row in range(sheet.nrows):
            parsed_row = []
            for column in range(sheet.ncols):
                parsed_row.append(sheet.cell_value(row, column))
            data.append(parsed_row)

            sentence = sheet.cell_value(row, 0)
            if sentence.startswith("#"):
                copy_row = [c for c in parsed_row]
                copy_row[0] = parsed_row[1]
                data.append(copy_row)

        workbook_new = xlwt.Workbook()
        new_sheet = workbook_new.add_sheet('test')
        for row_index, row_values in enumerate(data):
            del row_values[1] #remove unnecessary item
            for col_index, cell_value in enumerate(row_values):

                new_sheet.write(row_index, col_index, cell_value)
        workbook_new.save(out_file)


if __name__ == "__main__":
    modify_whyper_data("/home/huseyin/Desktop/Security/data/whyper/Read_Contacts.xls",
                       "/home/huseyin/Desktop/Security/data/whyper/Read_Contacts_modified.xls")
