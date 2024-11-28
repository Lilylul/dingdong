create database melon_chart;

use melon_chart;

create table chart(
	id int auto_increment primary key,
    `rank` int not null,
    title varchar(255) not null,
    artist varchar(255) not null);
    