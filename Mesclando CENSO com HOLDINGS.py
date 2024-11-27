import pandas as pd
import os
import time
from datetime import datetime
import logging
import sys
from typing import Tuple, Dict
import humanize
from tqdm import tqdm
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processamento_censo.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class CensoProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.censo_path = os.path.join(base_path, 'CENSO_EDUCACAO_SUPERIOR.csv')
        self.ies_path = os.path.join(base_path, 'ies.CSV')
        self.output_path = os.path.join(base_path, 'CENSO_EDUCACAO_SUPERIOR_COMPLETO.csv')
        self.start_time = None
        self.end_time = None

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info().rss
        return humanize.naturalsize(memory_info)

    def validate_files(self) -> bool:
        for file_path in [self.censo_path, self.ies_path]:
            if not os.path.exists(file_path):
                logging.error(f"Arquivo não encontrado: {file_path}")
                return False
            file_size = os.path.getsize(file_path)
            logging.info(f"Arquivo {os.path.basename(file_path)}: {humanize.naturalsize(file_size)}")
        return True

    def read_censo_file(self) -> pd.DataFrame:
        try:
            logging.info("Iniciando leitura do arquivo do Censo...")
            logging.info(f"Memória em uso antes da leitura: {self.get_memory_usage()}")
            
            # Definindo os nomes das colunas que precisamos
            cols_rename = {
                'NU_ANO_CENSO': 'NU_ANO_CENSO',
                'CO_IES': 'CO_IES'
            }
            
            total_lines = sum(1 for _ in open(self.censo_path, 'r', encoding='latin1'))
            chunks = []
            chunk_size = 100000
            
            with tqdm(total=total_lines, desc="Lendo arquivo do Censo") as pbar:
                for chunk in pd.read_csv(
                    self.censo_path,
                    sep=';',
                    encoding='latin1',
                    dtype=str,
                    chunksize=chunk_size,
                    quoting=1  # Alterado para QUOTE_ALL devido ao formato do arquivo
                ):
                    chunks.append(chunk)
                    pbar.update(len(chunk))
            
            df = pd.concat(chunks, ignore_index=True)
            
            logging.info(f"Arquivo do Censo lido com sucesso. Dimensões: {df.shape}")
            logging.info(f"Colunas encontradas: {df.columns.tolist()}")
            logging.info(f"Memória em uso após leitura: {self.get_memory_usage()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Erro ao ler arquivo do Censo: {str(e)}")
            raise

    def read_ies_file(self) -> pd.DataFrame:
        try:
            logging.info("Iniciando leitura do arquivo de IES...")
            
            # Definindo os nomes das colunas que sabemos que existem no arquivo
            cols = ['CO_IES', 'NO_IES', 'SG_IES', 'HOLDING', 'SG_IES2', 'INDEX']
            
            df = pd.read_csv(
                self.ies_path,
                sep=';',
                encoding='latin1',
                dtype=str,
                names=cols,  # Usando os nomes das colunas definidos
                header=0     # Primeira linha contém os cabeçalhos
            )
            
            logging.info(f"Arquivo de IES lido com sucesso. Dimensões: {df.shape}")
            logging.info(f"Colunas encontradas: {df.columns.tolist()}")
            
            return df
        except Exception as e:
            logging.error(f"Erro ao ler arquivo de IES: {str(e)}")
            raise

    def prepare_dataframes(self, censo_df: pd.DataFrame, ies_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            logging.info("Preparando DataFrames para merge...")
            start_prep = time.time()
            
            # Limpeza dos códigos CO_IES
            censo_df['CO_IES'] = censo_df['CO_IES'].astype(str).str.strip()
            ies_df['CO_IES'] = ies_df['CO_IES'].astype(str).str.strip()
            
            # Removendo espaços e caracteres especiais dos valores
            censo_df['CO_IES'] = censo_df['CO_IES'].str.replace('"', '').str.strip()
            ies_df['CO_IES'] = ies_df['CO_IES'].str.replace('"', '').str.strip()
            
            logging.info(f"Tempo de preparação: {time.time() - start_prep:.2f} segundos")
            logging.info(f"Memória em uso após preparação: {self.get_memory_usage()}")
            
            return censo_df, ies_df
            
        except Exception as e:
            logging.error(f"Erro na preparação dos DataFrames: {str(e)}")
            raise

    def merge_dataframes(self, censo_df: pd.DataFrame, ies_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        logging.info("Iniciando processo de merge...")
        start_merge = time.time()
        
        try:
            # Realizando o merge
            merged_df = pd.merge(
                censo_df,
                ies_df,
                on='CO_IES',
                how='left',
                indicator=True
            )
            
            # Coletando estatísticas
            merge_stats = {
                'total_records': len(merged_df),
                'matched_records': len(merged_df[merged_df['_merge'] == 'both']),
                'unmatched_records': len(merged_df[merged_df['_merge'] == 'left_only']),
                'unique_ies_censo': censo_df['CO_IES'].nunique(),
                'unique_ies_holdings': ies_df['CO_IES'].nunique(),
                'merge_time': time.time() - start_merge
            }
            
            # Removendo a coluna indicadora de merge
            merged_df.drop('_merge', axis=1, inplace=True)
            
            return merged_df, merge_stats
            
        except Exception as e:
            logging.error(f"Erro durante o merge: {str(e)}")
            raise

    def save_output(self, df: pd.DataFrame) -> None:
        logging.info("Iniciando salvamento do arquivo...")
        start_save = time.time()
        
        chunk_size = 100000
        total_rows = len(df)
        
        with tqdm(total=total_rows, desc="Salvando arquivo") as pbar:
            for i in range(0, total_rows, chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                mode = 'w' if i == 0 else 'a'
                header = i == 0
                
                chunk.to_csv(
                    self.output_path,
                    sep=';',
                    encoding='utf-8',
                    index=False,
                    mode=mode,
                    header=header,
                    quoting=1  # Alterado para QUOTE_ALL para manter consistência
                )
                pbar.update(len(chunk))
        
        save_time = time.time() - start_save
        logging.info(f"Arquivo salvo com sucesso em: {self.output_path}")
        logging.info(f"Tempo de salvamento: {save_time:.2f} segundos")

    def generate_report(self, merge_stats: Dict) -> None:
        self.end_time = time.time()
        processing_time = self.end_time - self.start_time
        
        output_size = os.path.getsize(self.output_path)
        
        report = [
            "\n=== RELATÓRIO DE PROCESSAMENTO ===",
            f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nEstatísticas de Processamento:",
            f"- Tempo total de execução: {processing_time:.2f} segundos ({processing_time/60:.2f} minutos)",
            f"- Tempo de merge: {merge_stats['merge_time']:.2f} segundos",
            f"- Total de registros processados: {merge_stats['total_records']:,}",
            f"- Registros com correspondência: {merge_stats['matched_records']:,}",
            f"- Registros sem correspondência: {merge_stats['unmatched_records']:,}",
            f"\nEstatísticas de IES:",
            f"- IES únicas no Censo: {merge_stats['unique_ies_censo']:,}",
            f"- IES únicas no arquivo de holdings: {merge_stats['unique_ies_holdings']:,}",
            f"\nInformações do Arquivo de Saída:",
            f"- Caminho: {self.output_path}",
            f"- Tamanho: {humanize.naturalsize(output_size)}",
            f"\nInformações de Sistema:",
            f"- Uso de memória final: {self.get_memory_usage()}",
            f"- CPU cores: {psutil.cpu_count()}",
            f"- Memória total do sistema: {humanize.naturalsize(psutil.virtual_memory().total)}"
        ]
        
        report_str = "\n".join(report)
        logging.info(report_str)
        
        with open('relatorio_processamento.txt', 'w', encoding='utf-8') as f:
            f.write(report_str)

    def process(self) -> bool:
        try:
            self.start_time = time.time()
            
            if not self.validate_files():
                return False
            
            censo_df = self.read_censo_file()
            ies_df = self.read_ies_file()
            
            censo_df, ies_df = self.prepare_dataframes(censo_df, ies_df)
            
            merged_df, merge_stats = self.merge_dataframes(censo_df, ies_df)
            
            self.save_output(merged_df)
            
            self.generate_report(merge_stats)
            
            logging.info("Processamento concluído com sucesso!")
            return True
            
        except Exception as e:
            logging.error(f"Erro durante o processamento: {str(e)}")
            return False

if __name__ == "__main__":
    BASE_PATH = r"C:\Dados\CENSO"
    
    try:
        processor = CensoProcessor(BASE_PATH)
        success = processor.process()
        sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Erro fatal durante o processamento: {str(e)}")
        sys.exit(1)