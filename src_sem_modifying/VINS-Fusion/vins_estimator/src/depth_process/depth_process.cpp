#include "depth_process.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
using namespace chrono;
using namespace cv;
using namespace std;
// using namespace nc;
 
Depth_process::Depth_process(/* args */){};
Depth_process::~Depth_process(){};
int Depth_process::k_means(cv::Mat srcImage)
{
	if (!srcImage.data)
	{
		printf("could not load image...\n");
		return -1;
	}
	imshow("depth.jpg", srcImage);
 
	//五个颜色，聚类之后的颜色随机从这里面选择
	Scalar colorTab[] = {
		Scalar(0,0,255),
		Scalar(0,255,0),
		Scalar(255,0,0),
		Scalar(0,255,255),
		Scalar(255,0,255)
	};
 
	int width = srcImage.cols;//图像的宽
	int height = srcImage.rows;//图像的高
	int channels = srcImage.channels();//图像的通道数
 
	//初始化一些定义
	int sampleCount = width*height;//所有的像素
	int clusterCount = 6;//分类数
	cv::Mat points(sampleCount, channels, CV_32F, Scalar(10));//points用来保存所有的数据
	cv::Mat labels;//聚类后的标签
	cv::Mat center(clusterCount, 1, points.type());//聚类后的类别的中心

	//将图像的RGB像素转到到样本数据,深度图只有单通道
	int index;
	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			index = i*width + j;
			// Vec3b bgr = srcImage.at<Vec3b>(i, j);
			 ushort depth = srcImage.at<ushort>(i, j);
			//将图像中的每个通道的数据分别赋值给points的值
			// points.at<float>(index, 0) = static_cast<int>(bgr[0]);
			// points.at<float>(index, 1) = static_cast<int>(bgr[1]);
			// points.at<float>(index, 2) = static_cast<int>(bgr[2]);
			points.at<float>(index, 0) = static_cast<int>(depth);
		}
	}
	//运行K-means算法
	//MAX_ITER也可以称为COUNT最大迭代次数，EPS最高精度,10表示最大的迭代次数，0.1表示结果的精确度
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT,10,0.1);
	kmeans(points, clusterCount, labels, criteria, 3, KMEANS_PP_CENTERS, center);
 
	//显示图像分割结果
	cv::Mat result = Mat::zeros(srcImage.size(),CV_8UC3 );//创建一张结果图 原本第二项是srcImage.type()
	for (int i = 0; i < srcImage.rows; i++)
	{
		for (int j = 0; j < srcImage.cols; j++)
		{
			index = i*width + j;
			int label = labels.at<int>(index);//每一个像素属于哪个标签
			result.at<Vec3b>(i, j)[0] = colorTab[label][0];//对结果图中的每一个通道进行赋值
			result.at<Vec3b>(i, j)[1] = colorTab[label][1];
			result.at<Vec3b>(i, j)[2] = colorTab[label][2];
		}
	}
	imshow("Kmeans", result);
	//连续显示选择
	waitKey(10); 
	return 0;
}

cv::Mat Depth_process::dep_k_means(cv::Mat srcImage){
	if (!srcImage.data)
	{
		printf("could not load image...\n");
		return cv::Mat(srcImage.size(), CV_8UC1, cv::Scalar(0));
	}
	// imshow("depth.jpg", srcImage);
 
	// //五个颜色，聚类之后的颜色随机从这里面选择
	// Scalar colorTab[] = {
	// 	Scalar(0,0,255),
	// 	Scalar(0,255,0),
	// 	Scalar(255,0,0),
	// 	Scalar(0,255,255),
	// 	Scalar(255,0,255)
	// };
	int width = srcImage.cols;//图像的宽
	int height = srcImage.rows;//图像的高
	int channels = srcImage.channels();//图像的通道数
 
	//初始化一些定义
	int sampleCount = width*height;//所有的像素
	int clusterCount = 6;//分类数
	cv::Mat points(sampleCount, channels, CV_32F, Scalar(10));//points用来保存所有的数据
	cv::Mat labels;//聚类后的标签
	cv::Mat center(clusterCount, 1, points.type());//聚类后的类别的中心

	//将图像的RGB像素转到到样本数据,这里是深度图，只有单通道unshort 16位转到CV_32
	int index;
	ushort *p_img;
	uchar *p_tar;
	// cout<<srcImage.type()<<"channels:"<<srcImage.channels()<<endl;
	for (int i = 0; i < srcImage.rows; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		for (int j = 0; j < srcImage.cols; j++)
		{
			index = i*width + j;
			// Vec3b bgr = srcImage.at<Vec3b>(i, j);
			//将图像中的每个通道的数据分别赋值给points的值
			// points.at<float>(index, 0) = static_cast<int>(bgr[0]);
			// points.at<float>(index, 1) = static_cast<int>(bgr[1]);
			// points.at<float>(index, 2) = static_cast<int>(bgr[2]);
			ushort depth = p_img[j];
			points.at<float>(index, 0) = static_cast<int>(depth);
		}
	}
	//运行K-means算法
	//MAX_ITER也可以称为COUNT最大迭代次数，EPS最高精度,10表示最大的迭代次数，0.1表示结果的精确度
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT,10,0.1);
	kmeans(points, clusterCount, labels, criteria, 1, KMEANS_PP_CENTERS,center);
 
	//显示图像分割结果
	cv::Mat target(srcImage.size(),CV_8UC1,Scalar(0));

	// int centertar = srcImage.rows / 2 * width + srcImage.cols / 2 ;   //中点
	//记录六个颜色的点个数，序号0-5
	vector<pair<int,int>> countcolors ;
	for (int i = 0;i<clusterCount;i++){
		countcolors.push_back(make_pair(i,0));
	}
	// for (int i = 0;i<centers.rows;i++){
	// 	int x = center.at<float>(i,0);
	// 	int y = center.at<float>(i,1);
	// }聚类中心
	// cv::Mat result = Mat::zeros(srcImage.size(),CV_8UC3 );//创建一张结果图 原本第二项是srcImage.type()
	for (int i = 0; i <srcImage.rows ; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		p_tar = target.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols ; j++)
		{
			index = i*width + j;
			// int label = labels.at<int>(index);//每一个像素属于哪个标签
			// result.at<Vec3b>(i, j)[0] = colorTab[label][0];//对结果图中的通道进行赋值
			// result.at<Vec3b>(i, j)[1] = colorTab[label][1];
			// result.at<Vec3b>(i, j)[2] = colorTab[label][2];

			countcolors.at(labels.at<int>(index)).second += 1; //统计个数
		}
	}
	//按点个数排序,升序
	sort(countcolors.begin(),countcolors.end(),[](const pair<int,int> &a,const pair<int,int>&b){return a.second>b.second;} );

	// for (int s=0;s<6;s++){
	// 	cout<<s<<":----"<<countcolors.at(s).first<<"----"<<countcolors.at(s).second<<endl;
	// }
	//再遍历一遍变色，或者可以使用map存下同色点的集合，可以稍微加速
	for (int i = 0; i <srcImage.rows ; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		p_tar = target.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols ; j++)
		{
			index = i*width + j;
			if( labels.at<int>(index) == countcolors.front().first){//与图像块中心位置像素同标签的mask保持255,其他改为0
				p_tar[j] = static_cast<int>(255);
			}
		}
	}
	// imshow("Kmeans", target);
	// waitKey(10);   //0.01s
	//需要返回的点的范围，尝试使用标签矩阵代替
	return target;
}

cv::Mat Depth_process::dep_contours(cv::Mat srcImage){
	if (!srcImage.data)
	{
		printf("could not load image...\n");
		return cv::Mat(srcImage.size(), CV_8UC1, cv::Scalar(255));
	}
	cv::Mat target(srcImage.size(),CV_8UC1 ,Scalar(255));
	//srcImage 是16UC1的格式
	// imshow("ori",srcImage);
	// waitKey(10);

	//变换图像格式
	auto start   = system_clock::now(); //计时开始
	int width = srcImage.cols;//图片宽度
	int height = srcImage.rows;//图片高度
	Mat dst = Mat::zeros(height, width, CV_8UC1);//先生成空的目标图片
	double minv = 0.0, maxv = 0.0;
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(srcImage, minp, maxp);  //取得像素值最大值和最小值

	//用指针访问像素，速度更快
	ushort *p_img;
	uchar *p_dst;
	for (int i = 0; i < height; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		p_dst = dst.ptr<uchar>(i);
		for (int j = 0; j < width; ++j)
		{
			p_dst[j] = (p_img[j] - minv) / (maxv - minv) * 255;

			//下面是失真较大的转换方法
			//int temp = img.at<ushort>(i, j);
			//dst.at<uchar>(i, j) = temp;
		}
	}
	
	imshow("8bit image", dst);
	waitKey(0);

	Mat canny_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;
	cv::Canny(dst,canny_output,100,200,3,false);
	findContours(canny_output, contours,hierachy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE,Point(0,0));
	cv::Mat ddst = Mat::zeros(srcImage.size(),CV_8UC3);
	RNG rng(12345);
	for (size_t i = 0;i<contours.size();i++){
		Scalar color = Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
		drawContours(ddst,contours,i,color,2,8,hierachy,0,Point(0,0));
	}
	auto end   = system_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout <<  "findcontours花费了"     << double(duration.count()) * microseconds::period::num / microseconds::period::den     << "秒" << endl;

	imshow("output",ddst);
	waitKey(10);
	return target;
}

cv::Mat Depth_process::dep_afkmc2(cv::Mat srcImage){
	if (!srcImage.data)
	{
		return cv::Mat(srcImage.size(), CV_8UC1, cv::Scalar(0));
	}

	int width = srcImage.cols;//图像的宽
	int height = srcImage.rows;//图像的高
	int channels = srcImage.channels();//图像的通道数
 
	//初始化一些定义
	int sampleCount = width*height;//所有的像素
	int clusterCount = 6;//分类数
	cv::Mat points(sampleCount, channels, CV_32F, Scalar(10));//points用来保存所有的数据


	//将图像的RGB像素转到到样本数据,这里是深度图，只有单通道unshort 16位转到CV_32
	int index;
	ushort *p_img;
	uchar *p_tar;
	// cout<<srcImage.type()<<"channels:"<<srcImage.channels()<<endl;
	for (int i = 0; i < srcImage.rows; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		for (int j = 0; j < srcImage.cols; j++)
		{
			index = i*width + j;
			ushort depth = p_img[j];
			points.at<float>(index) = static_cast<int>(depth);
		}
	}

	cv::Mat labels;//聚类后的标签
	cv::Mat center(clusterCount, 1, points.type());//聚类后的类别的中心
	// auto st   = system_clock::now();
	labels = afkmc2(points ,clusterCount, 100);
	// auto ed   = system_clock::now();
	// auto duration = duration_cast<microseconds>(ed - st);
	// cout <<  "初始点准备花费了"     << double(duration.count()) * microseconds::period::num / microseconds::period::den     << "秒" << endl;
	//运行K-means算法
	//MAX_ITER也可以称为COUNT最大迭代次数，EPS最高精度,10表示最大的迭代次数，0.1表示结果的精确度
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT,10,0.1);
	kmeans(points, clusterCount, labels, criteria,1, KMEANS_USE_INITIAL_LABELS,center);
	//显示图像分割结果
	cv::Mat target(srcImage.size(),CV_8UC1,Scalar(0));

	// int centertar = srcImage.rows / 2 * width + srcImage.cols / 2 ;   //中点
	//记录六个颜色的点个数，序号0-5
	vector<pair<int,int>> countcolors ;
	for (int i = 0;i<clusterCount;i++){
		countcolors.push_back(make_pair(i,0));
	}
	// for (int i = 0;i<centers.rows;i++){
	// 	int x = center.at<float>(i,0);
	// 	int y = center.at<float>(i,1);
	// }聚类中心
	// cv::Mat result = Mat::zeros(srcImage.size(),CV_8UC3 );//创建一张结果图 原本第二项是srcImage.type()
	for (int i = 0; i <srcImage.rows ; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		p_tar = target.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols ; j++)
		{
			index = i*width + j;
			// int label = labels.at<int>(index);//每一个像素属于哪个标签
			// result.at<Vec3b>(i, j)[0] = colorTab[label][0];//对结果图中的通道进行赋值
			// result.at<Vec3b>(i, j)[1] = colorTab[label][1];
			// result.at<Vec3b>(i, j)[2] = colorTab[label][2];

			countcolors.at(labels.at<int>(index)).second += 1; //统计个数
		}
	}
	//按点个数排序,升序
	sort(countcolors.begin(),countcolors.end(),[](const pair<int,int> &a,const pair<int,int>&b){return a.second>b.second;} );
	// for (int s=0;s<6;s++){
	// 	cout<<s<<":----"<<countcolors.at(s).first<<"----"<<countcolors.at(s).second<<endl;
	// }
	//再遍历一遍变色，或者可以使用map存下同色点的集合，可以稍微加速
	for (int i = 0; i <srcImage.rows ; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		p_tar = target.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols ; j++)
		{
			index = i*width + j;
			if( labels.at<int>(index) == countcolors.front().first){//与图像块中心位置像素同标签的mask保持255,其他改为0
				p_tar[j] = static_cast<int>(255);
			}
		}
	}
	// imshow("Kmeans", target);
	// waitKey(10);   //0.01s
	return target;
}

cv::Mat Depth_process::afkmc2(cv::Mat Xary , int k, int m){
	// cout<< s << "==="<<s.cols <<"=="<<s.rows<<"=="<<s.size()<<"=="<<s.str()<<endl;
	// [307200, 1]===1==307200==307200==[307200, 1]
	// return  cv::Mat(100, 100, CV_8UC1, cv::Scalar(255));
	srand((unsigned)(time(0)));
	std::vector<float> cv_X = Xary ;
	std::vector<int> ps_center(k);
	std::vector<float> cv_center(k);
	for(uint it=0 ;it<cv_center.size();it++){
		cv_center[it] = 0;
		ps_center[it] = 0;
	}
	cv::Mat label = cv::Mat(cv_X.size(),1,CV_32S);

	int r = rand() % (cv_X.size());
	cv_center[0] = cv_X[r];
	ps_center[0] = r;
	label.at<int>(r) = 0;

	// nc::NdArray<float> d_store = nc::empty<float>(cv_X.size(),k);
	// d_store.fill(-1);
	std::vector<std::vector<float>> d_store(cv_X.size(), std::vector<float>(k, -1));
	
	std::vector<float> cv_d2(cv_X.size());
	float d2sum=0.0f;
	for (uint i = 0; i < cv_X.size();i++){
		cv_d2[i] = distance(i,0,d_store,cv_X,cv_center);
		// cout<<cv_d2[i]<<endl;
		d2sum+=cv_d2[i];
	}
	// auto d2 = nc::NdArray<float>(cv_d2,false);
	std::vector<float> q(cv_X.size());
	float qsum = 0.0f;
	std::vector<float> q_rank(cv_X.size());
	// cout  <<"qsize---:"<<q.size()<<"    d2.sum:--=="<<d2sum<<"=----="<<accumulate(cv_d2.begin(),cv_d2.end(),0)<<endl;
	for (uint it =0; it < cv_d2.size();it++){
		q[it] = cv_d2[it]/ (2.0f*d2sum) + 1.0f/((2.0f)*cv_d2.size());
		qsum += q[it];
		if(it==0){
			q_rank[it] = q[it];
		}
		else{
			q_rank[it] = q[it] + q_rank[it-1];
		}
	} 
	// cout<<"center0: "<<r<<endl;
	float dx2 = MAXFLOAT;
    for (int i = 1;i<k;i++){
        int x = randomHit(q,qsum,q_rank);
		// cout<<"x :"<<x<<endl;
		for (int t=0 ;t<i; t++){
			float rs = distance(x,t,d_store,cv_X,cv_center);
			if (rs<dx2){
				dx2 = rs;
			}
		}
		float dy2 = MAXFLOAT;
		for (int j = 1;j<m;j++){
			int y = randomHit(q,qsum,q_rank);
			for (int t=0 ;t<i; t++){
				float rsy = distance(y,t,d_store,cv_X,cv_center);
				if (rsy<dy2){
					dy2 = rsy;
				}
			}
			float val = (float)(rand()) / RAND_MAX;
			if (dx2*q[y]==0 || (dy2*q[x])/(dx2*q[y])>val){
				x = y;
				dx2 = dy2;
			}
		}
    	cv_center[i] = cv_X[x];
		ps_center[i] = x;
		label.at<int>(x) = i;
		// cout<<"center"<<i<<": "<<x<<endl;
	}
	for(uint i= 0;i<cv_X.size();i++){
		int min = 256;
		int minid = 0;
		for(uint j = -1;j<cv_center.size();++j){
			int temp = cv_X[i] - cv_center[j];
			if(temp < min){
				min = temp ;
				minid = j;
			}
		}
		label.at<int>(i) = label.at<int>(ps_center[minid]);
	}
	// auto label = cv::Mat(cv_center.size(),1,CV_32S,cv_center.data());
	return label;
}

float Depth_process::distance(int i,int j,std::vector<std::vector<float>>&d_store,std::vector<float> &X,std::vector<float> &center){
	if (d_store[i][j]<0){
		// d_store(i,j) = cv::norm(X[i] - center[j]) * cv::norm(X[i]- center[j]);
		d_store[i][j] = X[i]-center[j];
	}
	return d_store[i][j];
}

int Depth_process::randomHit(const vector<float> &oddsList ,float &sum,const vector<float> &rank){
	float temp = (rand() %(10000+1))*0.0001f;
	float random = temp*(sum - 0.0f) + 0.0f;
	int len = oddsList.size();
	if(rank[0]>random) return 0;
	if(rank[len - 1]<random) return len-1;
	int left = 0;
	int right = len-1;
	int mid;
	while(left<right){
		mid = left+(right-left)/2;
		if(rank[mid]<random) left = mid+1;
		else right = mid;
	}
	return left;
 }


 cv::Mat Depth_process::ratiopro(cv::Mat srcImage){
	if (!srcImage.data)
	{
		return cv::Mat(srcImage.size(), CV_8UC1, cv::Scalar(0));
	}
	//将图片转化为数组
	int index = 0; 
	ushort mindep= 0, maxdep =65535;
	int width = srcImage.cols;//图像的宽
	int height = srcImage.rows;//图像的高
	//初始化一些定义
	int sampleCount = width*height;//所有的像素
	ushort *p_img;
	uchar *p_tar;
	vector<float>  data(sampleCount) ;
	// vector<float>data_ratio(sampleCount,0.f);
	cv::Mat target(srcImage.size(),CV_8UC1,Scalar(0));

	for (int i = 0; i <srcImage.rows ; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		p_tar = target.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols ; j++)
		{
			index = i*width + j;
			if(srcImage.at<ushort>(i,j)==0){
				data[index] = static_cast<int>(1);
			}
			else
				data[index] = (float)srcImage.at<ushort>(i,j);
			 //转数组
			// printf("%d \n", srcImage.at<ushort>(i,j));
		}
	}

	//比例的方法
	//一维数组排序
	// std::sort(data.begin(),data.end());
	// data_ratio[0] = 1;
	// for (uint i = 1;i< data.size();i++){
	// 	data_ratio[i] = data[i] / data[i-1];
	// }
	// float r_limit = 6;
	// int i = 1;
	// int j = 0;
	// int max_element = 1;
	// int i_max = 1;
	// int j_max = 0;
	// for(uint k =1 ;k<data.size();k++){
	// 	if(data_ratio[k]<=r_limit){
	// 		j = j+1;
	// 		if((j-i+1)>=max_element){
	// 			max_element = j-i+1;
	// 			i_max = i;
	// 			j_max = j;
	// 		}
	// 	}
	// 	else{
	// 		i = k;
	// 		j = k;
	// 	}
	// }
	// mindep = data[i_max];
	// maxdep = data[j_max-1];

	// cout<< mindep << "..."<<maxdep<<"..."<<i<<"..."<<j<<endl;

	//persistence1d方法
	// Persistence1D p;
	// p.RunPersistence(data);

	// //Get all extrema with a persistence larger than 10.
	// vector< TPairedExtrema > Extrema;
	// p.GetPairedExtrema(Extrema,10000);


	for (int i = 0; i <srcImage.rows ; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		p_tar = target.ptr<uchar>(i);
		for (int j = 0; j < srcImage.cols ; j++)
		{
			index = i*width + j;
			if( p_img[j] <=  maxdep && p_img[j] >= mindep){//与图像块中心位置像素同标签的mask保持255,其他改为0
				p_tar[j] = static_cast<int>(255);
			}
		}
	}
	return target;
 }

 cv::Mat Depth_process::dep_region_grow(cv::Mat srcImage){
	if (!srcImage.data)
	{
		return cv::Mat(srcImage.size(), CV_8UC1, cv::Scalar(0));
	}

	int width = srcImage.cols;//图像的宽
	int height = srcImage.rows;//图像的高
	int channels = srcImage.channels();//图像的通道数
 
	int sampleCount = width*height;//所有的像素
	cv::Mat points(sampleCount, channels, CV_32F, Scalar(10));//points用来保存所有的数据

	//将图像的RGB像素转到到样本数据,这里是深度图，只有单通道unshort 16位转到CV_32
	int index;
	ushort *p_img;
	// uchar *p_tar;
	for (int i = 0; i < srcImage.rows; i++)
	{
		p_img = srcImage.ptr<ushort>(i);//获取每行首地址
		for (int j = 0; j < srcImage.cols; j++)
		{
			index = i*width + j;
			ushort depth = p_img[j];
			points.at<float>(index) = static_cast<int>(depth);
		}
	}

	cv::Mat target(srcImage.size(),CV_8UC1,Scalar(0));
	//通过points或srcImage 划分投票得到生长初始点，阈值定为value
	int value = 200;
	cv::Point firstseed;
	//16等分，取中间4格再九等分，各自取中点，九点投票取得初始点
	//1-9 点分布设置如下：
	//6 2 7
	//3 1 4
	//8 5 9
	vector<pair<cv::Point,int>> NineP;
	NineP.push_back(make_pair(cv::Point(width/2,height/2),1));
	NineP.push_back(make_pair(cv::Point(width/2,height/2-height/6),1));
	NineP.push_back(make_pair(cv::Point(width/2-width/6,height/2),1));
	NineP.push_back(make_pair(cv::Point(width/2+width/6,height/2),1));
	NineP.push_back(make_pair(cv::Point(width/2,height/2+height/6),1));
	NineP.push_back(make_pair(cv::Point(width/2-width/6,height/2-width/6),1));
	NineP.push_back(make_pair(cv::Point(width/2+width/6,height/2-width/6),1));
	NineP.push_back(make_pair(cv::Point(width/2-width/6,height/2+width/6),1));
	NineP.push_back(make_pair(cv::Point(width/2+width/6,height/2+width/6),1));
	//按阈值投票
	for(uint pts = 0; pts<NineP.size();pts++){
		for(uint j = pts+1; j<NineP.size();j++){
			if ( abs(points.at<float>(NineP[pts].first.x *width +NineP[pts].first.y ) - points.at<float>(NineP[j].first.x *width +NineP[j].first.y) < value) ){
				NineP[pts].second +=1;
				NineP[j].second +=1 ;
			}
		}
	}
	//按点个数排序,升序
	sort(NineP.begin(),NineP.end(),[](const pair<cv::Point,int> &a,const pair<cv::Point,int>&b){return a.second>b.second;} );
	// cout<< "max point vote num is "<< NineP.front().second <<" and value :"<< points.at<float>(NineP.front().first.x *width +NineP.front().first.y)<<endl;
	firstseed = NineP.front().first;

	target = Region_Growing(points,target,firstseed,value);
	// imshow("Region_growing", target);
	// cvWaitKey(10);
	return target;
 }

cv::Mat Depth_process::Region_Growing(cv::Mat input, cv::Mat output, cv::Point fristseed, int value)
{
	vector<cv::Point>allseed;              //保存种子
	allseed.push_back(fristseed);         //压入第一个种子

	int direction[8][2] = { { -1, -1 }, { 0, -1 }, { 1, -1 }, { 1, 0 }, { 1, 1 }, { 0, 1 }, { -1, 1 }, { -1, 0 } };//种子选取的顺序
	// output = Mat::zeros(input.size(), CV_8UC1);               //创建一个黑图

	output.at<uchar>(fristseed.x, fristseed.y) = 255;         //第一个种子点设置为
	
	int cerseedvalue = 0;     //初始种子点的值

	int nextseedvalue = 0;    //与种子点相比较的下一点的值

	cv::Point comparseed;       //与种子点相比较的下一点
	
	cerseedvalue = input.at<float>(fristseed.x * output.cols + fristseed.y);    // 种子点的深度值

	while (!allseed.empty())
	{
		fristseed = allseed.back();   //最后一个种子点的
		allseed.pop_back();           //弹出最后一个种子点

		for (int i = 0; i < 8; ++i)
		{
			comparseed.x = direction[i][0] + fristseed.x;
			comparseed.y = direction[i][1] + fristseed.y;

			if (comparseed.x<0 || comparseed.x>(output.cols-1) || comparseed.y<0 || comparseed.y>(output.rows-1)){
				// cout<< input.size() <<endl;
				// cout<<"out of region "<<endl;
				continue;
			}
			
			if (output.at<uchar>(comparseed.x, comparseed.y) == 0)   //判断有没有被使用过
			{
				nextseedvalue = input.at<float>(comparseed.x * output.cols + comparseed.y);       //生长点的值
				if (abs(nextseedvalue - cerseedvalue) < value)//生长规则
				{
					// printf("value dist is %d \n",abs(nextseedvalue - cerseedvalue));
					allseed.push_back(comparseed);
					output.at<uchar>(comparseed.x, comparseed.y) =255;
				}
			}
		}
	}
	return output;
}