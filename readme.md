This project requires a file called `reviews.csv' with all the reviews data to analyse.
It's not commited to safeguard the data.

You can generate it in SQL with:
```
select translation_title, translation_text, score
from hotel_reputation;
```

As of today 2024/07/29 this returns 371448 rows. We export them to a file, the format can be CSV 
(with quotes to avoid comma conflicts) or JSON. JSON file weights 118 MB, while CSV only 93,7 MB, 
this is because JSON syntax introduces overhead. Since there is no reason to use JSON (we don't have 
nested data for instance) we will settle for CSV.
