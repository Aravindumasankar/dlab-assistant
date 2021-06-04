import pandas as pd
import os
import sys
import subprocess
import json
import glob
from PyPDF2 import PdfFileReader
import numpy as np
import errno

# Make a copy of the environment
env = dict(os.environ)
env['JAVA_OPTS'] = 'foo'
JAVA_NOT_FOUND_ERROR = "`java` command is not found from this Python process. Please ensure Java is installed and PATH is set for `java`"


# ==============================================================================


def _convert_pandas_csv_options(pandas_options, columns):
    ### This function is from tabula wrapper class ###
    ''' Translate `pd.read_csv()` options into `pd.DataFrame()` especially for header.

    Args:
        pandas_option (dict):
            pandas options like {'header': None}.
        columns (list):
            list of column name.
    '''

    _columns = pandas_options.pop('names', columns)
    header = pandas_options.pop('header', None)
    pandas_options.pop('encoding', None)

    if header == 'infer':
        header_line_number = 0 if not bool(_columns) else None
    else:
        header_line_number = header

    return _columns, header_line_number


def _extract_from(raw_json, pandas_options=None):
    ### This function is from tabula wrapper class ###
    '''Extract tables from json.

    Args:
        raw_json (list):
            Decoded list from tabula-java JSON.
        pandas_options (dict optional):
            pandas options for `pd.DataFrame()`
    '''

    data_frames = []
    if pandas_options is None:
        pandas_options = {}

    columns = pandas_options.pop('columns', None)
    columns, header_line_number = _convert_pandas_csv_options(pandas_options, columns)

    for table in raw_json:
        list_data = [[np.nan if not e['text'] else e['text'] for e in row] for row in table['data']]
        _columns = columns

        if isinstance(header_line_number, int) and not columns:
            _columns = list_data.pop(header_line_number)
            _columns = ['' if e is np.nan else e for e in _columns]

        data_frames.append(pd.DataFrame(data=list_data, columns=_columns, **pandas_options))

    return data_frames


class JavaNotFoundError(Exception):
    pass


def run_tabula(fullfilepath, java_options=[], options=['-p', 'all', '-g', '-f', 'JSON'], ):
    JAR_NAME = "tabula-1.0.3-jar-with-dependencies.jar"
    # JAR_DIR = 'C:\\Users\\212579645\\Downloads\\tabula-win-1.2.1\\tabula'
    JAR_DIR = 'C:/Users/212579645/PerfectWell/PW-Alpha/ETL/lib'
    JAR_PATH = os.path.join(JAR_DIR, JAR_NAME)
    args = ["java"] + java_options + ["-jar", JAR_PATH] + options + [fullfilepath]
    if not os.path.exists(fullfilepath):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fullfilepath)

    try:
        print
        args
        output = subprocess.check_output(args, shell=True)

    except FileNotFoundError as e:
        raise JavaNotFoundError(JAVA_NOT_FOUND_ERROR)

    except subprocess.CalledProcessError as e:
        sys.stderr.write("Error: {}\n".format(e.output.decode(encoding)))
        raise
    return output


try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


def main_guess():
    print
    "main"
    datadir = "C:/Users/212579645/Documents/Norway/Norway Oseberg DDRs/Oseberg Ost/NO 30 6 22"
    pdffiles = glob.glob(datadir + "/*.pdf")
    j2 = run_tabula(pdffiles[0])
    print
    len(j2)


def main_template():
    scriptout = ''
    datadir = "C:\\Users\\212579645\\Documents\\Norway\\Norway Oseberg DDRs\\Oseberg Ost\\NO 30 6 22"
    pdffiles = glob.glob(datadir + "\\*.pdf")
    for apdffile in pdffiles:
        pdf = PdfFileReader(open(apdffile, 'rb'))
        numpages = pdf.getNumPages()
        for apage in range(1, numpages + 1):
            linxufile = apdffile.replace('\\', '/')
            scriptout = scriptout + 'java -jar tabula-1.0.3-jar-with-dependencies.jar -a 37.515,22.583,62.773,570.082 -p ' + str(
                apage) + ' "' + linxufile + '" >>nordata/table_0__page_' + str(apage) + '.txt \n'
            scriptout = scriptout + 'java -jar tabula-1.0.3-jar-with-dependencies.jar -a 64.259,21.841,806.391,570.082 -p ' + str(
                apage) + ' "' + linxufile + '" >>nordata/table_1__page_' + str(apage) + '.txt \n'
            scriptout = scriptout + 'java -jar tabula-1.0.3-jar-with-dependencies.jar -a 811.591,22.583,831.649,570.082 -p ' + str(
                apage) + ' "' + linxufile + '" >>nordata/table_2__page_' + str(apage) + '.txt \n'
    fff = open("C:\\Users\\212579645\\Downloads\\tabula-win-1.2.1\\tabula\\" + 'test_norway.sh', 'w')
    fff.write(scriptout)
    fff.close()


def identify_and_convert_tables(df, outpath, tfile):
    print
    len(df)
    for i in range(len(df)):
        mydf = df[i]
        try:
            print
            mydf.shape
            mydf.dropna(axis=0, how='all', inplace=True)
            mydf.dropna(axis=1, how='all', inplace=True)
            print
            mydf.shape
        except:
            print
            '---------' + str(len(mydf))
        ss = mydf.to_string()
        if ('Time\\rFrom ' in ss) and ('Time\\rTo' in ss) and ('Hrs\\rUsed' in ss) and ('Activity Code' in ss):
            print
            i
            mydf.to_excel(outpath + '/' + tfile + '_Table_Operation_' + str(i) + '.xlsx')
            mydf.to_csv(outpath + '/' + tfile + '_Table_Operation_' + str(i) + '.csv')
            odf = mydf[~pd.isnull(mydf[mydf.columns[0]])]
            odf.dropna(axis=0, how='all', inplace=True)
            odf.dropna(axis=1, how='all', inplace=True)
            newlineLocs = odf.iloc[:, :5].applymap(lambda x: str(x).find('\n') > 0)
            odf.to_excel(outpath + '/' + tfile + '_Table_Operation_' + str(i) + '_modified.xlsx')
            odf.to_csv(outpath + '/' + tfile + '_Table_Operation_' + str(i) + '_modified.csv')

            # break
        if ('TD of Well at 24:00 MD ' in ss) and ('TD of Well at 24:00 TVD' in ss):
            print
            i
            mydf.to_excel(outpath + '/' + tfile + '_Table_DailyStatus_' + str(i) + '.xlsx')
        # if (('Water Based' in ss) or ('Oil Based' in ss)) and ('Funnel visc (s/l)' in ss):
        if ((' Based' in ss) or ('Sample depth' in ss)) and ('Funnel visc (s/l)' in ss):
            print
            i
            mydf.to_excel(outpath + '/' + tfile + '_Table_DrillingFluids_' + str(i) + '.xlsx')
        if ('Depth' in ss) and ('MD' in ss) and ('m TVD') and ('Pore pressure' in ss) and ('Reading' in ss):
            print
            i
            mydf.to_excel(outpath + '/' + tfile + '_Table_Pressure_' + str(i) + '.xlsx')

        if ('Group / Formation' in ss) and ('Code' in ss) and ('TVD'):
            print
            i
            mydf.to_excel(outpath + '/' + tfile + '_Table_FormationTop_' + str(i) + '.xlsx')

        if ('Aborted Operation' in ss) and ('Activity Code' in ss) and ('Failure Code'):
            print
            i
            mydf.to_excel(outpath + '/' + tfile + '_Table_Incident_' + str(i) + '.xlsx')


if __name__ == '__main__':
    datadir = os.path.join("C:\\", "Users", "212579645", "data", "Norway Oseberg DDRs")
    outpath = "C:\\Users\\212579645\\data\\output\\"
    import fnmatch
    import os

    pdffilesfound = []
    for root, dirnames, filenames in os.walk(datadir):
        for filename in fnmatch.filter(filenames, '*.pdf'):
            pdffilesfound.append(os.path.join(root, filename))
    for afile in pdffilesfound:
        tfile = afile.split('\\')[-1]
        print
        afile
        # print afile.replace('\',"/")
        print
        afile
        print
        tfile
        print
        'XXXXXXXXXXXXXXX'
        output = run_tabula(afile)
        output = output.replace(r"\r\n", r"\n")
        output = output.replace(r"\r", r"\n")
        output.decode('latin-1').encode("utf-8")
        encoding = 'ISO-8859-1'
        j1 = ''
        j1 = json.loads(output.decode(encoding))
        df = _extract_from(j1)
        identify_and_convert_tables(df, outpath, tfile)
        break

