# ASKLE

## To Test:

1. Create new env.<br>
``` diff
+ conda create -n askle python==3.10
```
3. Install [Docker](https://docs.docker.com/desktop/setup/install/mac-install/)
4. Makre sure docker-compose is working <br>
```diff
+ docker-compose -v
```
6. Install requirements<br>
```diff
+ pip install --no-cache-dir -r ./backend/requirements.txt
```
8. Run Docker (Desktop)
9. Build the app<br>
```diff
+ docker-compose up --build
```
11. ☕
12. Open browser, check the website at:<br>
```diff
+ http://localhost:3000/
````

## To Develop:
Please create on your own branch and pull from `main` periodically to get the latest update. <br>
```diff 
+ git checkout -b new-branch-name
```
<br>
<br>
![framework](https://github.com/user-attachments/assets/8e2d85ff-7767-4850-84c9-67f5e0b62080)
<br>
<br>


<sup><i>© Tiange Cui</i> 2025. All rights reserved. This software is not open source.</sup>
