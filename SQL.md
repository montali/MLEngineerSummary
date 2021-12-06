# SQL

First of all: keep in mind that SQL keywords are not case sensitive (but use upper-case as it's nicer).

## SELECT

Most important instruction, selects data from a database and returns a result table.

```sql
SELECT column1, column2, ...
FROM table_name;
```

### Examples

```sql
SELECT CustomerName, City, Country FROM Customers;
```

```sql
SELECT * FROM Customers;
```

`SELECT DISTINCT` is used to avoid returning duplicates.

## WHERE

`WHERE` filters records for `SELECT`, `UPDATE`, `DELETE`...

### Examples

```sql
SELECT * FROM Customers
WHERE Country = 'Mexico';
```

Note the usage of single quotes for text values.

```sql
SELECT * FROM Customers
WHERE CustomerID = 1;
```

The following operators can be used:

- `=`,`>`, `<`, `>=`, `<=`, `<>` (which is `!=`)
- `BETWEEN` to test in a range
- `LIKE` to search for patterns: `%` represents zero, one, multiple characters, `_` represents a single character
  - Example: `WHERE CustomerName LIKE 'a%'` finds names that start with an `a`
  - Example: `WHERE CustomerName LIKE '%or%'` finds any values containing "or" as substring
- `IN` to specify multiple values

The `WHERE` clause can be combined with `AND`, `OR`, `NOT`. You can use parenthesis to group the instructions.

## ORDER BY

The `ORDER BY` keyword is used to sort the result-set. By default, it sorts in **ascending order**, but you can have descending order by appending the `DESC` keyword.

## INSERT INTO

The `INSERT INTO` statement is used to insert new records in a table.

```sql
INSERT INTO table_name (column1, column2, column3, ...)
VALUES (value1, value2, value3, ...);
```

If you're adding values for all the column names, you can skip specifying them:

```sql
INSERT INTO table_name
VALUES (value1, value2, value3, ...);
```

### Examples

```sql
INSERT INTO Customers (CustomerName, ContactName, Address, City, PostalCode, Country)
VALUES ('Cardinal', 'Tom B. Erichsen', 'Skagen 21', 'Stavanger', '4006', 'Norway');
```

## IS NULL

To test for null values, you can use the `IS NULL` or `IS NOT NULL` keywords.

## UPDATE

The `UPDATE` statement is used to modify the existing records in a database.

```sql
UPDATE table_name
SET column1 = value1, column2 = value2, ...
WHERE condition;
```

Note that if you omit the `WHERE` clause, all records will be updated!

### Examples

```sql
UPDATE Customers
SET ContactName = 'Alfred Schmidt', City = 'Frankfurt'
WHERE CustomerID = 1;
```

## DELETE

The `DELETE` keyword can be used to delete existing records

```sql
DELETE FROM table_name WHERE condition;
```

Notice that if you forget the `WHERE` condition, you'll delete everything.

## LIMIT

The `LIMIT` clause is used to only return a given number of records. 

## MIN and MAX

The `MIN()` function returns the smallest value of the stated column, the `MAX()` the maximum.

```sql
SELECT MIN(column_name)
FROM table_name
WHERE condition;
```

### Examples

```sql
SELECT MIN(Price) AS SmallestPrice
FROM Products;
```

## COUNT, AVG, SUM

The `COUNT` statement returns the number of rows that match the criterion.

```sql
SELECT COUNT(column_name)
FROM table_name
WHERE condition;
```

The `AVG` statement returns the average value of a numeric column.

```sql
SELECT AVG(column_name)
FROM table_name
WHERE condition;
```

The `SUM` statement returns the total sum of a numeric column.

```sql
SELECT SUM(column_name)
FROM table_name
WHERE condition;
```

## AS

Aliases are used to give a table or a column a temporary name, in order to make things more readable. They only exist in the query.

### Examples

```sql
SELECT CustomerName, CONCAT_WS(', ', Address, PostalCode, City, Country) AS Address
FROM Customers;
```

```sql
SELECT o.OrderID, o.OrderDate, c.CustomerName
FROM Customers AS c, Orders AS o
WHERE c.CustomerName='Around the Horn' AND c.CustomerID=o.CustomerID;
```

## UNION

`UNION` combines two or more result-sets from `SELECT`s. The columns have to be similar. `UNION ALL` returns duplicates too.

```sql
SELECT City FROM Customers
UNION
SELECT City FROM Suppliers
ORDER BY City;
```

## GROUP BY

The `GROUP BY` statement groups rows that have the same values into summary rows, and it's extremely powerful when coupled with aggregate functions like `COUNT()`.

### Examples

Count the customers per each country.

```sql
SELECT COUNT(CustomerID), Country
FROM Customers
GROUP BY Country;
```

## HAVING

The `HAVING` keyword serves as a `WHERE` for aggregate functions.

### Examples

Return the count of people in countries, but only those having 5 or more people.

```sql
SELECT COUNT(CustomerID), Country
FROM Customers
GROUP BY Country
HAVING COUNT(CustomerID) > 5;
```

## EXISTS

The `EXISTS` operator tests for the existence of any record in a subquery.

### Examples

List suppliers with a product price less than 20.

```sql
SELECT SupplierName
FROM Suppliers
WHERE EXISTS (SELECT ProductName FROM Products WHERE Products.SupplierID = Suppliers.supplierID AND Price < 20);
```

## ANY, ALL

The `ANY` operator yields TRUE if any of the subquery values meet the condition, `ALL` yields TRUE if all the values meet the condition. Note that `EXISTS` didn't use an operator.

```sql
SELECT column_name(s)
FROM table_name
WHERE column_name operator ANY
  (SELECT column_name
  FROM table_name
  WHERE condition);
```

### Examples

```sql
SELECT ProductName
FROM Products
WHERE ProductID = ANY
  (SELECT ProductID
  FROM OrderDetails
  WHERE Quantity = 10);
```

```sql
SELECT ProductName
FROM Products
WHERE ProductID = ALL
  (SELECT ProductID
  FROM OrderDetails
  WHERE Quantity = 10);
```

