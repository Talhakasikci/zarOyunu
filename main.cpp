#include <iostream>
#include <ctime>//time fonksiyon prototipini icerir
#include <cstdlib>//srand ve rand fonksiyonlari icin gerekli prototipleri icerir

/*BU OYUNDA OYUNCU EN BASTA IKI ZAR ATAR. EGER BU IKI ZAR TOPLAMI 7 VE YA 11 İSE OYUNCU KAZANIR.
 * EGER BU SAYILARIN TOPLAMI 2,3 YA DA 12 ISE OYUNCU BU OYUNU KAYBEDER.
 * EGER TOPLAM 4,5,6,8,9, YA DA 10 İSE OYUN DEVAM EDER VE BU DEGER OYUNCUNUN PUANI OLUR.
 * ZARLARI BIR SONRAKI ATISINDA AYNI PUANI BULAMSI GEREKIR. EGERBU ESNADA 7 GELIRSE OYUNCU KAYBEDER*/


using namespace std;

unsigned int zarAt();

int main(){
    cout<<"Zar oyununa hosgeldiniz"<<endl;
    cout<<endl;//baslik ile oyun arasi bosluk
        
    enum Durum{KAZANDI,KAYBETTI,DEVAM};
    
    srand(static_cast<unsigned int>(time(0)));
    
    unsigned int puanim = 0;//oyunun devam etmesi durumunda oyuncunun puanini tutar
    Durum oyunDurumu = KAZANDI;
    unsigned int zarToplam = zarAt();//ilk zar atisi
     

     
     switch(zarToplam){//switch baslangici
         
         case 7:
         case 11:
         oyunDurumu=KAZANDI;
         break;
         
         case 2:
         case 3:
         case 12:
         oyunDurumu=KAYBETTI;
         break;
         
         default:
         oyunDurumu=DEVAM;
         puanim = zarToplam;
         cout<<"Puaniniz : "<<puanim<<endl;
         break;
         
     }//switch bitisi
     

     
          while(oyunDurumu==DEVAM){//oyun durumu DEVAM geldigi surece devam edicek
        zarToplam = zarAt();//tekrar zar at
        
        if(zarToplam == puanim){
            oyunDurumu=KAZANDI;
        }
        else if(oyunDurumu==7){
            oyunDurumu=KAYBETTI;
        }
         
     }//while bitis
     
     if(oyunDurumu==KAZANDI){
         cout<<"\nOyun kazanildi"<<endl;
     }else {
         cout<<"\Oyun kaybedildi"<<endl;
     }
     
     }//main sonu
     

    
     
    
unsigned int zarAt(){
    unsigned int zar1 = 1+rand()%6;//ilk zar atisi
    unsigned int zar2 = 1 + rand()%6;//ikinci zar atisi
    unsigned toplam = zar1+zar2;//iki zarin toplami
         
    cout<<"Zar durumu : "<<toplam<<endl;
         
    return toplam;
}

   
