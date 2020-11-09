---
title: Gobblin data Ingestion from Mysql to S3
tags: [Apache Gobblin, Aws S3, Big Data, MySQL]
style: fill
color: success
description: A unified data ingestion framework for extracting, transforming and loading large volume of data from variety of data sources. 
---

https://gobblin.readthedocs.io/en/latest/img/Gobblin-Logo.png

## Gobblin

Gobblin is unified data ingestion framework for extracting, transforming and loading large volume of data from variety of data sources. User can define job according to their requirements and get it done easily.
Below figure show the task flow of the gobblin job.

![[https://gobblin.readthedocs.io/en/latest/img/Gobblin-Constructs.png](https://gobblin.readthedocs.io/en/latest/img/Gobblin-Constructs.png)](https://cdn-images-1.medium.com/max/3460/0*Nsb4Hq1_qXHHMDJH.png)*[https://gobblin.readthedocs.io/en/latest/img/Gobblin-Constructs.png](https://gobblin.readthedocs.io/en/latest/img/Gobblin-Constructs.png)*

The main properties of the gobblin jobs are Source, Extractor, Converter and Publisher. User can also use Quality Checker, Fork operator and Writer properties in gobblin job.
This post will guide for data ingestion from MySQL to s3.

First download and add [aws-java-sdk-\<version>.jar](https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/2.8.5/hadoop-aws-2.8.5.jar) and [hadoop-aws-\<version>.jar](https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk/1.11.80/aws-java-sdk-1.11.80.jar) to `gobblin-dist/lib/`.

## MySQL

Create sample database and add some data within it. Below example creates employees database with only single table named employees.

    mysql> CREATE DATABASE IF NOT EXISTS employees; 
    mysql> USE employees; 
    mysql> CREATE TABLE employees ( 
        -> emp_no INT NOT NULL, 
        -> birth_date DATE NOT NULL,  
        -> first_name VARCHAR(14) NOT NULL, 
        -> last_name VARCHAR(16) NOT NULL, 
        -> gender ENUM ('M','F') NOT NULL, 
        -> hire_date DATE NOT NULL, 
        -> PRIMARY KEY (emp_no)); 
    mysql> INSERT INTO `employees` VALUES 
        -> (10001,'1953-09-02','Georgi','Facello','M','1986-06-26'), 
        -> (10002,'1964-06-02','Bezalel','Simmel','F','1985-11-21'), 
        -> (10003,'1959-12-03','Parto','Bamford','M','1986-08-28'), 
        -> (10004,'1954-05-01','Chirstian','Koblick','M','1986-12-01'),     
        -> (10005,'1955-01-21','Kyoichi','Maliniak','M','1989-09-12'), 
        -> (10006,'1953-04-20','Anneke','Preusig','F','1989-06-02'), 
        -> (10007,'1957-05-23','Tzvetan','Zielinski','F','1989-02-10'),    
        -> (10008,'1958-02-19','Saniya','Kalloufi','M','1994-09-15'), 
        -> (10009,'1952-04-19','Sumant','Peac','F','1985-02-18');

## Gobblin job

Create two folders named `job_config` and `work_dir`. Create two files in job_config directory named `job.properties` and `MysqlToS3.pull`. Here job.properties contains the common properties of the all pull files within its directory and subsquent directories and .pull file contains unique properties for perticular job.

As mentioned earlier job has mainlly four properties Source, Extractor, Converter and Publisher. Add source and sink configurations in properties files as below.
> *Find more details on gobblin job properties [here](https://gobblin.readthedocs.io/en/latest/user-guide/Configuration-Properties-Glossary/).*

## Source properties

Add source class and credentials as below. (Replace credentials)

    # Source properties - source class to extract data from Mysql Source source.class=org.apache.gobblin.source.extractor.extract.jdbc.MysqlSource 

    # Source properties 
    source.max.number.of.partitions=1 source.querybased.partition.interval=1 source.querybased.is.compression=false source.querybased.watermark.type=timestamp 

    # Source connection properties source.conn.driver=com.mysql.jdbc.Driver 
    source.conn.username=<username> 
    source.conn.password=<password> 
    source.conn.host=<hostname> 
    source.conn.port=<port> source.conn.timeout=1500

## Converter and Qualitychecker properties

Provide converter class and other properties. Here AVRO converter is used.

    # Converter properties - Record from mysql source will be processed by the below series of converters converter.classes=org.apache.gobblin.converter.avro.JsonIntermediateToAvroConverter 

    # date columns format 
    converter.avro.timestamp.format=YYYY-MM-DD HH:MM:SS converter.avro.date.format=yyyy-MM-dd converter.avro.time.format=HH:mm:ss 

    # Qualitychecker properties qualitychecker.task.policies=org.apache.gobblin.policies.count.RowCountPolicy,org.apache.gobblin.policies.schema.SchemaCompatibilityPolicy 
    qualitychecker.task.policy.types=OPTIONAL,OPTIONAL

## Writer and Publisher properties

Add output format, destination type, filesystem uri, etc properties.

    # Writer properties 
    writer.destination.type=HDFS 
    writer.output.format=AVRO 
    writer.fs.uri=s3a://<bucket-name>/employee/final writer.output.dir=s3a://<bucket-name>/employee/task-output writer.builder.class=org.apache.gobblin.writer.AvroDataWriterBuilder 

    state.store.fs.uri=s3a://<bucket-name>/employee/state state.store.dir=/employee/state-store task.data.root.dir=s3a://<bucket-name>/employee/workspace/task-staging 

    # Publisher properties data.publisher.type=org.apache.gobblin.publisher.BaseDataPublisher data.publisher.fs.uri=s3a://<bucket-name>/employee/ data.publisher.metadata.output.dir=s3a://<bucket-name>/employee/metadata_out 
    data.publisher.final.dir=s3a://<bucket-name>/employee/job-output 

    # metrics and mapreduce properties 
    mr.job.max.mappers=1 
    mr.job.root.dir=/employee/working metrics.reporting.file.enabled=true metrics.log.dir=/employee/metrics 
    metrics.reporting.file.suffix=txt

## Sink properties

Create new access key in aws, a buffer directory in local system and add below sink properties.
> *Find aws s3a configurations [here](https://hadoop.apache.org/docs/current/hadoop-aws/tools/hadoop-aws/index.html)*

    fs.s3a.access.key=<AWSAccessKeyId> 
    fs.s3a.secret.key=<AWSSecretKey> fs.s3a.buffer.dir=/path/to/buffer/dir 
    fs.s3a.endpoint=s3.<region>.amazonaws.com fs.s3a.path.style.access=true

Create MysqlToS3.pull file and add job properties, schema name, entity name, etc.
> *Keep only those .pull files in this directory which you want to run.*

## MysqlToS3.pull

    # job properties 
    job.name=MysqlToS3 
    job.group=movielense 
    job.description=pull data from mysql to S3 
    job.lock.enabled=flase 

    # Source properties 
    source.querybased.schema=movielense 
    source.entity=links 
    source.querybased.extract.type=snapshot 

    # Extract properties 
    extract.namespace=movielense 
    extract.table.type=snapshot_only 
    extract.table.name=links

Before starting the job start mysql server.

    $ sudo service mysql start 
    $ export GOBBLIN_JOB_CONFIG_DIR=/path/to/job_config 
    $ export GOBBLIN_WORK_DIR=/path/to/work_dir 

    # start job 
    $ gobblin.sh service standalone start --jars /gobblin/build/gobblin-sql/libs/gobblin-sql-0.15.0.jar 

To see the execution details open the `standalone.out` file in `/gobblin/build/gobblin-sql/logs/`. On completion of the job the `.pull` file will be renamed as `.pull.done`. Check out `s3a://<bucket-name>/employee/job-output`.
to see ingested data.
> ***References:***
[https://gobblin.readthedocs.io/en/latest/Getting-Started/](https://gobblin.readthedocs.io/en/latest/Getting-Started/)
[https://gobblin.readthedocs.io/en/latest/user-guide/Configuration-Properties-Glossary/](https://gobblin.readthedocs.io/en/latest/user-guide/Configuration-Properties-Glossary/)
[https://hadoop.apache.org/docs/current/hadoop-aws/tools/hadoop-aws/index.html](https://hadoop.apache.org/docs/current/hadoop-aws/tools/hadoop-aws/index.html)
[https://gobblin.readthedocs.io/en/latest/case-studies/Publishing-Data-to-S3/](https://gobblin.readthedocs.io/en/latest/case-studies/Publishing-Data-to-S3/)
[https://github.com/apache/incubator-gobblin/blob/f5b893cf3972a89c6557e7899138e34493882dcb/gobblin-example/src/main/resources/distcpToS3.job](https://github.com/apache/incubator-gobblin/blob/f5b893cf3972a89c6557e7899138e34493882dcb/gobblin-example/src/main/resources/distcpToS3.job)
