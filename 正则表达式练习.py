import re

# 1.写一个正则表达式，使其能同时识别下面所有的字符串：
# 示例输出
'''
['bat', 'bit', 'but', 'hat', 'hit', 'hut']
'''
data ="bat ,bit ,but ,hat ,hit ,hut"



# 2.匹配一行文字中所有开头的字母
# 示例输出
'''
['i', 'l', 'y', 'b', 'y', 'd', 't', 'l', 'm']
'''
data = 'i love you but you don\'t love me'



# 3.匹配阅读次数
# 示例输出
'''
9999
'''
data = '阅读次数为 9999'



# 4.统计出python、c、java相应⽂章阅读的次数
# 示例输出
'''
['2342', '7980', '9999']
'''
data = 'python = 2342,c = 7980,java = 9999'



# 5.匹配出163的邮箱地址，且@符号之前有4到20位，例如hello@163.com
# 示例输出
'''
aabbcc@163.com
'''
data = 'aabbcc@163.com, aabbcc123@qq.com'



# 6.提取每行中完整的年月日和时间字段
# 示例输出
'''
['1988-01-01 17:20:10', '2018-02-02 02:29:01']
'''
data="""time 1988-01-01 17:20:10 fsadf 2018-02-02 02:29:01"""



# 7.匹配由单个空格分隔的任意单词对，也就是姓和名
# 示例输出
'''
[('Han', 'meimei'), ('Li', 'lei'), ('Zhang', 'san'), ('Li', 'si')]
'''
data = 'Han meimei, Li lei, Zhang san, Li si'



# 8.提取图片的url
# 示例输出
'''
https://rpic.douyucdn.cn/appCovers/2016/11/13/1213973_201611131917_small.jpg
'''
data = """
    <img
    src="https://rpic.douyucdn.cn/appCovers/2016/11/13/1213973_201611131917_small.jpg"
    style="display:inline;">
"""



# 9.去掉后缀
# 示例输出
'''
http://www.interoem.com/
http://3995503.com/
http://lib.wzmc.edu.cn/
http://www.zy-ls.com/
http://www.fincm.com/
'''
data = """
http://www.interoem.com/messageinfo.asp?id=35
http://3995503.com/class/class09/news_show.asp?id=14
http://lib.wzmc.edu.cn/news/onews.asp?id=769
http://www.zy-ls.com/alfx.asp?newsid=377&id=6
http://www.fincm.com/newslist.asp?id=415
"""



# 10.爬取就业信息：
# 示例输出
''' 
    岗位职责：
    完成推荐算法、数据统计、接⼝、后台等服务器端相关⼯作
    必备要求： 良好的⾃我驱动⼒和职业素养，⼯作积极主动、结果导向     技术要求：
    1、⼀年以上  Python  开发经验，掌握⾯向对象分析和设计，了解设计模式
    2、掌握HTTP协议，熟悉MVC、MVVM等概念以及相关WEB开发框架
    3、掌握关系数据库开发设计，掌握      SQL，熟练使⽤   MySQL/PostgreSQL        中 的⼀种
    4、掌握NoSQL、MQ，熟练使⽤对应技术解决⽅案
    5、熟悉      Javascript/CSS/HTML5，JQuery、React、Vue.js
        加分项：
    ⼤数据，数理统计，机器学习，sklearn，⾼性能，⼤并发。
'''

data = """
<div>
   <p>岗位职责：</p>
   <p>完成推荐算法、数据统计、接⼝、后台等服务器端相关⼯作</p>
   <p><br></p> <p>必备要求：</p> <p>良好的⾃我驱动⼒和职业素养，⼯作积极主动、结果导向</p>   <p> <br></p> <p>技术要求：</p>
   <p>1、⼀年以上	Python	开发经验，掌握⾯向对象分析和设计，了解设计模式</p >
   <p>2、掌握HTTP协议，熟悉MVC、MVVM等概念以及相关WEB开发框架</p>
   <p>3、掌握关系数据库开发设计，掌握	SQL，熟练使⽤	MySQL/PostgreSQL	中 的⼀种<br></p>
   <p>4、掌握NoSQL、MQ，熟练使⽤对应技术解决⽅案</p>
   <p>5、熟悉	Javascript/CSS/HTML5，JQuery、React、Vue.js</p>
   <p> <br></p> <p>加分项：</p>
   <p>⼤数据，数理统计，机器学习，sklearn，⾼性能，⼤并发。</p>
</div>
"""



