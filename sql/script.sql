/* https://docs.microsoft.com/en-us/sql/t-sql/functions/lag-transact-sql?view=sql-server-ver15 */
/* Get the value the column before*/
SELECT col_1 AS column_one,   
       LAG(col_2, 1,0) OVER (ORDER BY YEAR(col_3)) AS column_two  
FROM table_one  
WHERE col_one = 275 and YEAR(col_3) IN ('2005','2006');

/* Compare values within partitions */
SELECT TerritoryName, BusinessEntityID, SalesYTD,   
       LAG (SalesYTD, 1, 0) OVER (PARTITION BY TerritoryName ORDER BY SalesYTD DESC) AS PrevRepSales  
FROM Sales.vSalesPerson  
WHERE TerritoryName IN (N'Northwest', N'Canada')   
ORDER BY TerritoryName;

/* Compare value from previous data*/
SELECT CalendarYear, CalendarQuarter, SalesAmountQuota AS SalesQuota,  
       LAG(SalesAmountQuota,1,0) OVER (ORDER BY CalendarYear, CalendarQuarter) AS PrevQuota,  
       SalesAmountQuota - LAG(SalesAmountQuota,1,0) OVER (ORDER BY CalendarYear, CalendarQuarter) AS Diff  
FROM dbo.FactSalesQuota  
WHERE EmployeeKey = 272 AND CalendarYear IN (2001, 2002)  
ORDER BY CalendarYear, CalendarQuarter; 

/* https://docs.microsoft.com/en-us/sql/t-sql/functions/lead-transact-sql?view=sql-server-ver15*/
/* Compare values between years */
SELECT BusinessEntityID, YEAR(QuotaDate) AS SalesYear, SalesQuota AS CurrentQuota,   
    LEAD(SalesQuota, 1,0) OVER (ORDER BY YEAR(QuotaDate)) AS NextQuota  
FROM Sales.SalesPersonQuotaHistory  
WHERE BusinessEntityID = 275 and YEAR(QuotaDate) IN ('2005','2006');

/* Compare values within partitions */
SELECT TerritoryName, BusinessEntityID, SalesYTD,   
       LEAD (SalesYTD, 1, 0) OVER (PARTITION BY TerritoryName ORDER BY SalesYTD DESC) AS NextRepSales  
FROM Sales.vSalesPerson  
WHERE TerritoryName IN (N'Northwest', N'Canada')   
ORDER BY TerritoryName;

/* Compare values between quarters*/
SELECT CalendarYear AS Year, CalendarQuarter AS Quarter, SalesAmountQuota AS SalesQuota,  
       LEAD(SalesAmountQuota,1,0) OVER (ORDER BY CalendarYear, CalendarQuarter) AS NextQuota,  
   SalesAmountQuota - LEAD(Sale sAmountQuota,1,0) OVER (ORDER BY CalendarYear, CalendarQuarter) AS Diff  
FROM dbo.FactSalesQuota  
WHERE EmployeeKey = 272 AND CalendarYear IN (2001,2002)  
ORDER BY CalendarYear, CalendarQuarter; 

/* PERCENTILE_CONT */
SELECT DISTINCT Name AS DepartmentName  
      ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ph.Rate)   
                            OVER (PARTITION BY Name) AS MedianCont  
      ,PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY ph.Rate)   
                            OVER (PARTITION BY Name) AS MedianDisc  
FROM HumanResources.Department AS d  
INNER JOIN HumanResources.EmployeeDepartmentHistory AS dh   
    ON dh.DepartmentID = d.DepartmentID  
INNER JOIN HumanResources.EmployeePayHistory AS ph  
    ON ph.BusinessEntityID = dh.BusinessEntityID  
WHERE dh.EndDate IS NULL;

/*PERCENTILE_DISC*/
SELECT DISTINCT Name AS DepartmentName  
      ,PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ph.Rate)   
                            OVER (PARTITION BY Name) AS MedianCont  
      ,PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY ph.Rate)   
                            OVER (PARTITION BY Name) AS MedianDisc  
FROM HumanResources.Department AS d  
INNER JOIN HumanResources.EmployeeDepartmentHistory AS dh   
    ON dh.DepartmentID = d.DepartmentID  
INNER JOIN HumanResources.EmployeePayHistory AS ph  
    ON ph.BusinessEntityID = dh.BusinessEntityID  
WHERE dh.EndDate IS NULL;

SELECT CONCAT(year, '-', month) AS "YYYY-MM", COUNT(*) AS view_cnt, COUNT(DISTINCT(user_id)) AS unique_user_cnt
FROM usergram.gram_time_based_partitioned_client_id_44
WHERE location_uri LIKE '%blog.usergram.info%'
    AND (CAST( year AS INTEGER), CAST( month AS INTEGER)) >= (2020, 10)
GROUP BY year, month
ORDER BY year, month

WHERE REGEXP_LIKE(location_url,'^app.usergram.info/users/[0-9,_,-]+')

SELECT AVG(NULLIF(COALESCE(current_year,  
   previous_year), 0.00)) AS 'Average Budget'  -- return null if both expressions are equal

SELECT * from table_name where column_name REGEXP/RLIKE '^[abcde]' -- the first letter contains abcde
SELECT * from table_name where column_name REGEXP/RLIKE '^[^abcde]' -- the first letter NOT contains abcde
SELECT * from table_name where column_name REGEXP/RLIKE '[abcde]$' -- the last letter contains abcde
SELECT * FROM mytable WHERE SUBSTRING(coliumn_name, -1) IN ('1', '2') -- last character in (1,2)

SELECT 
    CASE 
    WHEN A + B > C AND A+C>B AND B+C>A 
        THEN 
            CASE WHEN A = B AND B = C THEN 'Equilateral' 
                 WHEN A = B OR B = C OR A = C THEN 'Isosceles' 
                 WHEN A != B OR B != C OR A != C THEN 'Scalene' END
         ELSE 'Not A Triangle' END
    FROM TRIANGLES; 

select itemID
from t
group by itemId
having sum(case when categoryID = 10 then 1 else 0 end) > 0 and
       sum(case when categoryID = 16 then 1 else 0 end) > 0;

-- Function
SELECT DIV(5);
SELECT MOD(5);
SELECT TRIM('   remove the space  ');
SELECT FLOOR(35.4); -- return 35
SELECT EXP(5);
SELECT LOG(5);
SELECT LOG10(2);
SELECT POW(2),2; -- 2 power 2
SELECT GREATEST(2,3) -- return 3(highest)
SELECT LEAST(2,3) -- return 2(lowest)
SELECT RADIANS(8);
SELECT SQRT(196);
SELECT TRUNCATE(234.45,1); --return 234.4
SELECT RAND(); --random number
SELECT TOP 3; --select the top 3 of the record
SELECT CHAR_LENGTH(column);
SELECT LENGTH(column); --different with 'char_length' & 'length'
SELECT CHARACTER_LENGTH(column);
SELECT RIGHT('This is string', 6); --will return "string"
SELECT LEFT('This is string', 4); -- will return "This"
SELECT SUBSTR('This is string', 1,4); -- will return "This"
SELECT INSERT('Hello word', 7, 5, 'World'); --replace word with world


CREATE TABLE Persons (
    ID int NOT NULL,
    LastName varchar(255) NOT NULL,
    FirstName varchar(255),
    Age int,
    PRIMARY KEY (ID)
);

select count(sale_price) from table_name
select count(nvl(sale_price,0)) from table_name -- use this instead if its consist null value
IFNULL(), ISNULL(), COALESCE(), and NVL()
-- datetime mysql

ADDDATE("2017-06-15", INTERVAL 10 DAY); -- add 10 days to the current date
ADDTIME(datetime, addtime) -- example SELECT ADDTIME("09:34:21.000001", "2:10:5.000003"); add to current time
SELECT SUBDATE("2017-06-15", INTERVAL 10 DAY); --  subtract date from the current date

  COUNT(DISTINCT
    CASE
      WHEN
        date >= '2020-03-19' THEN table_name
      ELSE -- OR NULL 
        NULL
      END
    ) AS 'column_name'

    ROUND(
        PERCENT_RANK() OVER (
            PARTITION BY e.department_id
            ORDER BY salary
        ) 
    ,2) percentile_rank

SELECT DATE_FORMAT(subdate(datetime, weekday(datetime)), '%Y-%m-%d') AS weekly,
       COUNT(CASE
                 WHEN datetime >= '2020-06-22' THEN endpoint = '/api/filter/' 
                      OR NULL
             END) AS 'after release',
       COUNT(CASE
                 WHEN datetime < '2020-06-21' THEN endpoint = '/api/filter/' 
                      OR NULL
             END) AS 'before release'

-- json array
SELECT * ,JSON_EXTRACT(query, '$[0][1].activity_type') as activity_edge --https://jsonformatter.curiousconcept.com/ 
from table_name
where JSON_EXTRACT(query, '$[0][1][0].activity_type') = 1
                           
SELECT DATE_FORMAT(date, '%Y-%m') AS monthly,
    SUM(CASE WHEN app_user_usage.date < '2020-08-1' THEN app_user_usage.filter END) AS 'filter usage before release',
    SUM(CASE WHEN app_user_usage.date BETWEEN '2020-08-1' AND '2020-09-29' THEN app_user_usage.filter END) AS 'filter usage after release UK-238/UK-535',
    SUM(CASE WHEN app_user_usage.date >= '2020-09-30' THEN app_user_usage.filter END) AS 'filter usage after UK-266',
    SUM(CASE WHEN app_user_usage.date < '2020-08-1' THEN app_user_usage.observe END) AS 'observe usage before release',
    SUM(CASE WHEN app_user_usage.date BETWEEN '2020-08-1' AND '2020-09-29' THEN app_user_usage.observe END) AS 'observe usage after release UK-238/UK-535',
    SUM(CASE WHEN app_user_usage.date >= '2020-09-30' THEN app_user_usage.observe END) AS 'observe usage after UK-266'
FROM app_user_usage
    INNER JOIN app_user ON app_user_usage.app_user_id = app_user.id
WHERE app_user.login_name NOT LIKE '%bebit%'
    AND app_user.is_staff = 0
    AND app_user.client_id IN (
        SELECT client_id
        FROM be_def
        WHERE created_at < '2020-08-1'
    UNION
        SELECT client_id
        FROM contact_def
        WHERE created_at < '2020-08-1')
    AND app_user_usage.date >= '2020-01-6'
GROUP BY monthly
ORDER BY monthly DESC
                           
WITH filter_observe_total AS
  (SELECT app_user.client_id,
          SUM(CASE
                  WHEN app_user_usage.date < '2020-08-1' THEN app_user_usage.filter
              END) AS filter_usage_before_release,
          SUM(CASE
                  WHEN app_user_usage.date >= '2020-08-1' THEN app_user_usage.filter
              END) AS filter_usage_after_release,
          SUM(CASE
                  WHEN app_user_usage.date < '2020-08-1' THEN app_user_usage.observe
              END) AS observe_usage_before_release,
          SUM(CASE
                  WHEN app_user_usage.date >= '2020-08-1' THEN app_user_usage.observe
              END) AS observe_usage_after_release
   FROM app_user_usage
   INNER JOIN app_user ON app_user_usage.app_user_id = app_user.id
   WHERE app_user.login_name NOT LIKE '%bebit%'
     AND app_user.is_staff = 0
     AND app_user.client_id IN
       (SELECT client_id
        FROM be_def
        WHERE created_at < '2020-08-1'
        UNION SELECT client_id
        FROM contact_def
        WHERE created_at < '2020-08-1')
     AND app_user_usage.date >= '2020-01-6'
   GROUP BY app_user.client_id)
SELECT client_id,
       (filter_usage_after_release - filter_usage_before_release) AS filter_total,
       (observe_usage_after_release - observe_usage_before_release) AS observe_total
FROM filter_observe_total
ORDER BY filter_total DESC


-- link to jso sql array 
-- https://dev.mysql.com/doc/refman/8.0/en/json.html
-- https://dev.mysql.com/doc/refman/5.7/en/json-search-functions.html

