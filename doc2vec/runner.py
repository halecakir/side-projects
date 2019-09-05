from utils import Utils, OverallResultRow
import numpy
from nltk.tokenize import RegexpTokenizer
import xlwt
import datetime

import gensim.models as g
import codecs


if __name__ == '__main__':

    threshold = 0.50

    inputFolder = "IO_Input"
    outputFolder = "IO_Result"
    fileStamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    vectorFile = "enwiki_dbow/doc2vec.bin"
    chunk_gram = "Doc2Vec"

    # model
    model = "enwiki_dbow/doc2vec.bin"
    # inference hyper-parameters
    start_alpha = 0.01
    infer_epoch = 1000
    # load model
    m = g.Doc2Vec.load(model)

    logFileName = outputFolder + "/" + fileStamp + "_logs.txt"
    logFile = open(logFileName, "w")
    logFile.write("Vector File   -----------------------------------\n")
    logFile.write(vectorFile)
    logFile.write("\n")
    logFile.write("Chunk Gram   -----------------------------------\n")
    logFile.write(chunk_gram)
    logFile.write("\n")
    logFile.write("Threshold   -----------------------------------\n")
    logFile.write(str(threshold))
    logFile.write("\n")
    logFile.close()

    outputFileName = outputFolder + "/" + fileStamp + "_results.xls"
    outputFile = xlwt.Workbook()

    # whyper_result_files = ["Read_Calendar", "Read_Contacts", "Record_Audio"]
    whyper_result_files = ["Record_Audio"]

    for inputFileName in whyper_result_files:

        words, w2i, permissions, applications = Utils.read_whyper_data(inputFolder, inputFileName + ".xls", "excel", True, chunk_gram, False)

        sheet1 = outputFile.add_sheet(inputFileName)

        cols = ["Application ID", "Description", "Description Vector", "Manually-marked"]
        txt = "Row %s, Col %s"
        row = sheet1.row(0)
        for index, col in enumerate(cols):
            row.write(index, col)
        app_line_index = 1

        permissionWords = inputFileName.split("_")
        permissionVector = m.infer_vector(permissionWords, alpha=start_alpha, steps=infer_epoch)

        for current_app in applications:

            tokenizer = RegexpTokenizer(r'\w+')
            doc_words = tokenizer.tokenize(current_app.description)

            current_app.dsc_vec = m.infer_vector(doc_words, alpha=start_alpha, steps=infer_epoch)

            result = Utils.cos_similiariy(permissionVector, current_app.dsc_vec)
            man_marked = 0

            for current_dsc_sentence in current_app.dsc_sentences:
                if current_dsc_sentence.manual_marked == 1 :
                    man_marked = man_marked + 1

            # print ("------------------------------------------------------------------------------")
            # print ("- Application ID : " + current_app.app_id+"\n")
            # print ("- App Description : " + current_app.description+"\n\n")
            # print ("- Document Vector: " + str(len(current_app.dsc_vec)) + "\n")
            # print ("- Permission Vector: " + str(len(permissionVector)) + "\n")
            # print ("- Similarity: " + str(result) + "\n\n")

            # cols = ["Application ID", "Description", "Description Vector", "Manually-marked"]
            row = sheet1.row(app_line_index)
            row.write(0, current_app.app_id)
            row.write(1, current_app.description)
            # dsc_vec_string = numpy.array2string(current_app.dsc_vec)
            i = 0
            dsc_vec_string = ""
            for dsc_vec_val in current_app.dsc_vec:
                dsc_vec_string = dsc_vec_string + str(dsc_vec_val)
                if i != len(current_app.dsc_vec)-1 :
                    dsc_vec_string = dsc_vec_string + ", "
            row.write(2, dsc_vec_string)
            row.write(3, man_marked)

            # if result < 0:
            #     result = result - (2 * result)
            #
            # row.write(3, result)


            # # True Positive
            # if (man_marked == 1) & (result >= threshold):
            #     row.write(4, 1)
            # # False Positive
            # elif (man_marked == 0) & (result >= threshold):
            #     row.write(5, 1)
            # # False Negative
            # elif (man_marked == 1) & (result < threshold):
            #     row.write(6, 1)
            # # True Negative
            # elif (man_marked == 0) & (result < threshold):
            #     row.write(7, 1)
            # else:
            #     row.write(14, "XXX")

            app_line_index = app_line_index + 1

        # row = sheet1.row(app_line_index+2)
        # row.write(4, xlwt.Formula("SUM(E2:E"+str(app_line_index)+")"))
        # row.write(5, xlwt.Formula("SUM(F2:F"+str(app_line_index)+")"))
        # row.write(6, xlwt.Formula("SUM(G2:G"+str(app_line_index)+")"))
        # row.write(7, xlwt.Formula("SUM(H2:H"+str(app_line_index)+")"))
        #
        # row = sheet1.row(app_line_index + 4)
        #
        # row.write(4, "Precision")
        # row.write(5, "Recall")
        # row.write(6, "F-Score")
        # row.write(7, "Accuracy")
        #
        # TP = "E"+str(app_line_index+3)
        # FP = "F"+str(app_line_index+3)
        # FN = "G"+str(app_line_index+3)
        # TN = "H"+str(app_line_index+3)
        #
        # # write table row for current permission
        # row = sheet1.row(app_line_index + 5)
        #
        # # Precision --- TP / (TP + FP)
        # row.write(4, xlwt.Formula(TP+"/SUM("+TP+","+FP+")"))
        #
        # # Recall --- TP / (TP + FN)
        # row.write(5, xlwt.Formula(TP+"/SUM("+TP+","+FN+")"))
        #
        # # F-Score --- 2 x Precision x Recall / (Precision+Recall)
        # row.write(6, xlwt.Formula("PRODUCT(2,E"+str(app_line_index+6)+",F"+str(app_line_index+6)+") / SUM(E"+str(app_line_index+6)+",F"+str(app_line_index+6)+")"))
        #
        # # Accuracy --- row.write(9, xlwt.Formula( (TP + TN) / (TP + FP + TN + FN )
        # row.write(7, xlwt.Formula("SUM("+TP+","+TN+")/SUM( "+TP+","+FP+","+TN+","+FN+")"))


    sheet1 = outputFile.add_sheet("Comparison")

    # cols = ["Permission", "SI", "TP", "FP", "FN", "TN", "P(%)", "R(%)", "FS(%)", "Acc(%)"]
    # txt = "Row %s, Col %s"
    # row = sheet1.row(0)
    # for index, col in enumerate(cols):
    #     row.write(index, col)
    #
    # cols = ["READ CONTACTS", "204", "186", "18", "49", "2930", "91.2", "79.1", "84.7", "97.9"]
    # txt = "Row %s, Col %s"
    # row = sheet1.row(1)
    # for index, col in enumerate(cols):
    #     row.write(index, col)
    #
    # cols = ["READ CALENDAR", "288", "241", "47", "42", "2422", "83.7", "85.1", "84.4", "96.8"]
    # txt = "Row %s, Col %s"
    # row = sheet1.row(2)
    # for index, col in enumerate(cols):
    #     row.write(index, col)
    #
    # cols = ["RECORD AUDIO", "259", "195", "64", "50", "3470", "75.9", "79.7", "77.4", "97.0"]
    # txt = "Row %s, Col %s"
    # row = sheet1.row(3)
    # for index, col in enumerate(cols):
    #     row.write(index, col)
    #
    # cols = ["TOTAL", "751", "622", "129", "141", "9061", "82.8", "81.5", "82.2", "97.3"]
    # txt = "Row %s, Col %s"
    # row = sheet1.row(4)
    # for index, col in enumerate(cols):
    #     row.write(index, col)

    outputFile.save(outputFileName)
