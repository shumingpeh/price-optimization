
# coding: utf-8

# This notebook tries to illustrate the idea of how to proceed with the price optimization problem for kraft. My previous experience with EA, might not be entirely transferrable due to the nuances of the industry. Still, the theory is still applicable
# 
# There are 2 tiers towards the price optimization
# 1. Getting demand model for items in kraft's
# 1. From demand model, price optimization will be carried out

# # Components of analysis
# 1. Exploratory data analysis of kraft's data
# 1. Understanding kraft's seasonality, and transforming it to stationary
# 1. Demand generation modeling for each item in kraft's inventory
# 1. Price optimization of each item from demand generation
# 1. Forecast of daily sales from recommendation
# 1. Draw down of revenue from sales? Duration of recovery

# ## Exploratory data analysis of kraft's data

# - Here, it will be to fully understand the underlying data of kraft. There can be more upon having the data, but should at least comprise of the following:
#     - General idea of sales/demand trend of each cateogry
#         - if autocorrelation is present --> if sales today is high, would it follow or taper
#     - How often does sales occur, and when it does, how does it affect sales; during and after
#     - Inventory level, speed of replendisment
#     - If broken assortment effect is present in fmcg
#     - A good feel of the idea, how grainular it can be too (weekly, daily)

# ## Understanding kraft's seasonality, and transforming it to stationary

# - for each category of items, i would like to know the seasonality it faces
#     - so that adjustment can be applied appropriately for the demand model later 

# ## Demand generation modeling for each item in kraft's inventory

# - To give more context why choosing to model the demand first
#     - Here, we try to model out every kraft item ($Q_i$) in terms of all kraft item prices, competitor prices, if promotion is present or not and etc
#     - This would be give us more understanding of how demand of item is directly affected with the prices considered
# - The variables that go into the model might include the following (and more):
#     - is holiday period --> (binary, 1 or 0)
#     - is promotion present --> (binary, 1 or 0)
#     - is aisle placing --> (binary, 1 or 0)
#     - is offline advertising present --> (binary, 1 or 0)
#     - previous period demand 
#         - this is meant to introduce some idea of autocorrelation
#     - competitors price
#     - competitors sales numbers
#     - price of items
#     - broken assortment effect 
#         - might not be applicable
#     - age of item (how close to expiry date)
# - The demand model should look something like this:
#     - $Q_i$ = $\beta_0*IsHolidayPeriod$ + $\beta_1*IsPromotionPresent$ + $\beta_2*IsAislePlacing$ + $\beta_3*IsOfflineAdvertisingPresent$ + $\beta_4*PreviousPeriodDemand_i$ + $\beta_5*CompetitorPrice_i$ + $\beta_6*CompetitorQuantitySold_i$ + $\beta_7*KraftPrices_r$ + $\beta_8*BrokenAssortmentEffect$ +  $\beta_9*AgeOfItem_i$
#         - $Q_i$ = quantity sold for item

# ## Price optimization of each item from demand generation

# - The main motivation of having quantity (demand) in terms of prices, are as follows:
#     - it can be directly translated into revenue with $Q_i * P_i$
#     - it can be approached as a mathematical problem to optimize all of kraft prices
# - Constraints can be applied to the price optimization
#     - a max (in)decrease of prices of 20%
#     - a limit of number of items to optimize price for
# 
# ###### For an example of demand and price optimization, it can be referred to [link](https://github.com/shumingpeh/price-optimization)
# 

# ## Forecast of daily sales from recommendation

# - After the obtaining the price optimization results, the forecast of sales will be generated to obtain the amount of uplift
# - A pilot test can be set up, so as to see if the recommendations do follow through as expected

# ## Draw down of revenue from sales? Duration of recovery

# - If prices are not going to be permanent and changed temporary
# - How do demand (and revenue) get affected after the sales?
#     - We are able to quantify the recovery duration with a statistical theory: mean reversion
#     - past experience with EA (ping me for presentation, due to sensitive materials)
