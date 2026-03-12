FROM spark:3.5.8
USER root

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

USER spark

RUN set -ex && \
    # org.slf4j:slf4j-api:2.0.17
    curl -L https://repo1.maven.org/maven2/org/slf4j/slf4j-api/2.0.17/slf4j-api-2.0.17.jar \
        -o /opt/spark/jars/slf4j-api-2.0.17.jar && \
    chmod 644 /opt/spark/jars/slf4j-api-2.0.17.jar && \
    \
    # commons-codec:commons-codec:1.18.0
    curl -L https://repo1.maven.org/maven2/commons-codec/commons-codec/1.18.0/commons-codec-1.18.0.jar \
        -o /opt/spark/jars/commons-codec-1.18.0.jar && \
    chmod 644 /opt/spark/jars/commons-codec-1.18.0.jar && \
    \
    # com.google.j2objc:j2objc-annotations:3.0.0
    curl -L https://repo1.maven.org/maven2/com/google/j2objc/j2objc-annotations/3.0.0/j2objc-annotations-3.0.0.jar \
        -o /opt/spark/jars/j2objc-annotations-3.0.0.jar && \
    chmod 644 /opt/spark/jars/j2objc-annotations-3.0.0.jar && \
    \
    # org.apache.spark:spark-avro_2.12:3.5.5
    curl -L https://repo1.maven.org/maven2/org/apache/spark/spark-avro_2.12/3.5.5/spark-avro_2.12-3.5.5.jar \
        -o /opt/spark/jars/spark-avro_2.12-3.5.5.jar && \
    chmod 644 /opt/spark/jars/spark-avro_2.12-3.5.5.jar && \
    \
    # org.apache.hadoop:hadoop-aws:3.3.6
    curl -L https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.6/hadoop-aws-3.3.6.jar \
        -o /opt/spark/jars/hadoop-aws-3.3.6.jar && \
    chmod 644 /opt/spark/jars/hadoop-aws-3.3.6.jar && \
    \
    # org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.7.1
    curl -L https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-spark-runtime-3.5_2.12/1.7.1/iceberg-spark-runtime-3.5_2.12-1.7.1.jar \
        -o /opt/spark/jars/iceberg-spark-runtime-3.5_2.12-1.7.1.jar && \
    chmod 644 /opt/spark/jars/iceberg-spark-runtime-3.5_2.12-1.7.1.jar && \
    \
    # com.amazonaws:aws-java-sdk-bundle:1.12.780
    curl -L https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.780/aws-java-sdk-bundle-1.12.780.jar \
        -o /opt/spark/jars/aws-java-sdk-bundle-1.12.780.jar && \
    chmod 644 /opt/spark/jars/aws-java-sdk-bundle-1.12.780.jar && \
    \
    # org.apache.iceberg:iceberg-aws:1.7.1 (for S3FileIO)
    curl -L https://repo1.maven.org/maven2/org/apache/iceberg/iceberg-aws/1.7.1/iceberg-aws-1.7.1.jar \
        -o /opt/spark/jars/iceberg-aws-1.7.1.jar && \
    chmod 644 /opt/spark/jars/iceberg-aws-1.7.1.jar
