import os

nl = "\n"

def python_hive_sql(QUERY_DICT: dict):
    """ Generate the datasets from HIVE SQL - interface with the Command Line
    
    args:
    QUERY_DICT -> {OUTPUT_NAME: [DB_NAME, HIVE SQL QUERY]}
    
    return:
    prints the query, filename and filesize
    """
    
    for OUTPUT_NAME, QUERY in QUERY_DICT.items():
        # OUTPUT_NAME = fun_path_join(...)
        HADOOP_CLIENT_OPTS = "-Ddisable.quoting.for.sv=false"
        DB = f"jdbc:hive2://hdpe-hive-dr.sgp.dbs.com:10000/{QUERY[0]};principal=hive/hdpe-hive-dr.sgp.dbs.com@REG1.1BANK.DBS.COM"
        OPTIONS = "--incremental=true --outputformat=csv2 --silent=true"
        query = f"env HADOOP_CLIENT_OPTS='{HADOOP_CLIENT_OPTS}' beeline -u '{DB}' {OPTIONS} -e\"{QUERY[1]}\">{OUTPUT_NAME}"

        print(f"Beeline Query:{nl}{query}{nl}")
        
        os.system(query) # run query

        file_name = os.path.basename(OUTPUT_NAME)
        file_size = os.path.getsize(OUTPUT_NAME) / 1000_000
        
        print(f"{file_name} is {file_size}mb{nl}")
    
    return True
    
  
