FEATURE DOCUMENTATION 



This document describes all features included in the dataset for household electricity consumption modeling. The target variable is future\_6h\_consumption, representing the electricity consumption for the next 6 hours.





BASIC IDENTIFICATION AND TIME FEATURES

household\_ID: Unique identifier for each household.



DATE: Date of the meter reading (YYYY-MM-DD).



TIME: Time of the meter reading (HH:MM:SS).



timestamp: Combined DATE and TIME indicating the exact reading time.



hour: Hour of the day extracted from the timestamp (0–23).



dayofweek: Day of the week (0 = Monday, 6 = Sunday).



month: Month of the year (1–12).



### 2\. METER READING AND CONSUMPTION FEATURES



***TOTAL\_IMPORT (kWh):*** Cumulative electricity import reading from the smart meter.



***backward\_avg\_consumption***: Average consumption over a historical backward-looking window.



***avg\_5d\_same\_time\_kwh***: Average electricity consumption during the same time window (±2 hours) over the past 5 days.



***avg\_1d\_same\_time\_kwh***: Average electricity consumption during the same time window (±2 hours) on the previous day.



***avg\_1w\_same\_time\_kwh:*** Average electricity consumption during the same time window (±2 hours) on the same day of the previous week.



***consumption\_per\_member***: Electricity consumption normalized by the number of household members.



future\_6h\_consumption (TARGET):Target variable. Total electricity consumption during the next 6 hours after the current timestamp.



### 3\. HOUSEHOLD DEMOGRAPHIC CHARACTERISTICS (W1 SURVEY)



***w1\_hh\_member\_count:*** Total number of household members.



***w1\_hh\_avg\_age:*** Average age of household members.



***w1\_hh\_num\_children***: Number of children in the household.



***w1\_hh\_num\_seniors***: Number of senior members in the household.



***w1\_hh\_avg\_hours\_home***: Average number of hours household members spend at home per day.



***w1\_hh\_share\_went\_out\_for\_work:*** Proportion of household members who travel outside for work.



### 4\. APPLIANCE OWNERSHIP AND USAGE (FANS AND LIGHTING)



***w1\_num\_fans***: Number of fans in the household.



***w1\_fan\_hours\_day***: Average number of hours fans are used during daytime.



***w1\_fan\_hours\_night*:** Average number of hours fans are used during nighttime.



***w1\_num\_lights***: Number of lights in the household.



***w1\_light\_total\_wattage***: Total wattage of all lighting devices.



***w1\_light\_hours\_day***: Average number of hours lights are used during daytime.



***w1\_light\_hours\_night***: Average number of hours lights are used during nighttime.



***fan\_density***: Derived feature indicating the density of fans per room.





### 5\. HOUSING STRUCTURE AND PHYSICAL ATTRIBUTES



***w1\_num\_rooms***: Number of rooms in the house.



***w1\_total\_windows***: Total number of windows.



***w1\_total\_doors\_ext***: Number of exterior doors.



***w1\_total\_room\_bulbs***: Total number of bulbs installed in rooms.



***w1\_total\_room\_fans***: Total number of fans installed in rooms.



***w1\_total\_room\_acs***: Total number of air conditioners installed in rooms.



***w1\_num\_bedrooms***: Number of bedrooms in the household.



***own\_the\_house\_or\_living\_on\_rent***: Indicates whether the household owns the home or lives on rent.



***built\_year\_of\_the\_house***: Year the house was built.



***type\_of\_house***: Type of housing structure (e.g., detached, semi-detached, apartment).



***floor\_area***: Total floor area of the house (in square units).



### 6\. HOUSEHOLD ACTIVITIES AND SOCIOECONOMIC STATUS



***is\_there\_business\_carried\_out\_in\_the\_household:*** Indicates whether any business is run from the household.



***socio\_economic\_class:*** Socioeconomic classification of the household.



***total\_monthly\_expenditure\_of\_last\_month***: Total monthly household expenditure in the previous month.



***method\_of\_receiving\_water:*** How the household obtains water (e.g., piped water, tanker).



***water\_heating\_method\_for\_bathing:*** Method used to heat water for bathing.



***boil\_water\_before\_drinking:*** Indicates whether drinking water is boiled before consumption.



***no\_of\_times\_food\_cooked\_last\_week:*** Number of times food was cooked in the household in the last week.



### 7\. COOKING ENERGY SOURCES



***gas\_used\_for\_cooking***: Gas usage for cooking (binary or categorical).



***electricity\_from\_national\_grid\_used\_for\_cooking***: Whether electricity from the national grid is used for cooking.



***electricity\_generated\_using\_solar\_energy\_used\_for\_cooking***: Whether solar-generated electricity is used for cooking.



***firewood\_used\_for\_cooking***: Whether firewood is used for cooking.



***kerosene\_used\_for\_cooking***: Whether kerosene is used for cooking.



***sawdust\_or\_paddy\_husk\_used\_for\_cooking***: Whether sawdust or paddy husk is used for cooking.



***biogas\_used\_for\_cooking***: Whether biogas is used for cooking.



***coconut\_shells\_or\_charcoal\_used\_for\_cooking***: Whether charcoal or coconut shells are used for cooking.

