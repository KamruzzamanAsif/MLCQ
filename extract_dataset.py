import pandas as pd
import requests
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create codes folder
codes_folder = 'codes'
if not os.path.exists(codes_folder):
    os.makedirs(codes_folder)
    logger.info(f"Created folder: {codes_folder}")

# Load dataset
df = pd.read_csv('MLCQCodeSmellSamples.csv', sep=';', skipinitialspace=True)
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Process only first 10 rows
df = df.head(10)

def get_code_from_github_link(row, idx):
    """Extract code from GitHub using the link column"""
    try:
        smell = row['smell'].strip()
        repo = row['repository'].strip()
        link = row['link'].strip()
        start_line = int(row['start_line'])
        end_line = int(row['end_line'])
        
        logger.info(f"[{idx}] Processing: {smell} from {repo}")
        logger.info(f"[{idx}] Lines {start_line}-{end_line}")
        
        # Convert GitHub web link to raw GitHub link
        # From: https://github.com/owner/repo/blob/commit/path/#L1-L10
        # To: https://raw.githubusercontent.com/owner/repo/commit/path
        
        raw_link = link.replace('/blob/', '/raw/').split('/#')[0]
        logger.debug(f"[{idx}] Raw URL: {raw_link}")
        
        response = requests.get(raw_link, timeout=15)
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            # Extract lines (1-indexed)
            code_snippet = '\n'.join(lines[start_line-1:end_line])
            logger.info(f"[{idx}] ✓ SUCCESS - Retrieved {len(code_snippet)} chars")
            return code_snippet
        else:
            error_msg = f"HTTP {response.status_code}"
            logger.warning(f"[{idx}] ✗ FAILED - {error_msg}")
            return f"Error: {error_msg}"
            
    except Exception as e:
        logger.error(f"[{idx}] ✗ EXCEPTION - {str(e)}")
        return f"Error: {str(e)}"

# Add code column
logger.info(f"Starting extraction for {len(df)} rows...")
logger.info("=" * 80)

df['code'] = df.apply(lambda row: get_code_from_github_link(row, df.index.get_loc(row.name) + 1), axis=1)

logger.info("=" * 80)
logger.info(f"Extraction complete!")

# Save code snippets to individual files
logger.info(f"Saving code snippets to {codes_folder}/ folder...")
for idx, row in df.iterrows():
    code_id = str(row['id']).strip()
    code_content = row['code']
    file_path = os.path.join(codes_folder, f"{code_id}.java")
    
    try:
        with open(file_path, 'w') as f:
            f.write(code_content)
        logger.info(f"✓ Saved: {code_id}.java")
    except Exception as e:
        logger.error(f"✗ Error saving {code_id}.java: {str(e)}")

# Save with code
output_file = 'MLCQCodeSmellSamples_with_code.csv'
df.to_csv(output_file, index=False)
logger.info(f"Saved CSV to {output_file}")

# Show summary
success_count = (df['code'].str.startswith('Error:') == False).sum()
logger.info(f"Success: {success_count}/{len(df)} rows")
logger.info(f"Code files saved in: {codes_folder}/")
