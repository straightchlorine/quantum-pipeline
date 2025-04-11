#!/usr/bin/env python
"""
Script to be executed by Airflow for quantum feature processing.
This script calls the main functionality defined in the quantum processing module.
"""

import argparse
import logging
import sys
from datetime import datetime

# basic logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_processing')

# import the quantum processing module
try:
    from spark_scripts.quantum_incremental_processing import (
        DEFAULT_CONFIG,
        check_for_new_data,
        create_spark_session,
        list_available_topics,
        process_experiments_incrementally,
    )
except ImportError:
    logger.error(
        "Could not import the quantum processing module. Make sure it's in the PYTHONPATH."
    )
    sys.exit(1)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process quantum experiment data.')
    parser.add_argument('--s3_endpoint', default='http://server:9000', help='S3 endpoint URL')
    parser.add_argument(
        '--s3_bucket', default='s3a://local-vqe-results/experiments/', help='S3 bucket path'
    )
    parser.add_argument(
        '--s3_warehouse', default='s3a://local-features/warehouse/', help='S3 warehouse path'
    )
    parser.add_argument('--spark_master', default='spark://server:7077', help='Spark master URL')
    parser.add_argument('--host_ip', default='station', help='Host IP address')

    return parser.parse_args()


def main(config=None):
    """Main entry point for the script."""
    start_time = datetime.now()
    logger.info(f'Starting quantum processing at {start_time}')

    args = parse_args()

    # create spark session
    spark = create_spark_session()

    try:
        if config is None:
            config = DEFAULT_CONFIG

        bucket_path = config.get('S3_BUCKET', DEFAULT_CONFIG['S3_BUCKET'])
        available_topics = list_available_topics(spark, bucket_path)

        for topic in available_topics:
            print(f'Found topic: {topic}')

            # check for new data
            topic_name, df = check_for_new_data(spark, topic, config)

            if df is None:
                print('No new data to process.')
                return

            # Process data incrementally
            results = process_experiments_incrementally(spark, df, topic_name)
            summary = '\nProcessing Summary:\n' + '\n'.join(
                [f'{table}: {count} new records processed' for table, count in results.items()]
            )
            return summary
    except Exception as e:
        logger.error(f'Error during processing: {e}', exc_info=True)
        raise
    finally:
        end_time = datetime.now()
        logger.info(
            f'Quantum processing completed at {end_time}. Duration: {end_time - start_time}'
        )


if __name__ == '__main__':
    main()
