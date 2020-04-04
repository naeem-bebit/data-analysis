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