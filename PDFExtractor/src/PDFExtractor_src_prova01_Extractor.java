package prova01;
import org.apache.pdfbox.cos.COSBase;
import org.apache.pdfbox.cos.COSName;
import org.apache.pdfbox.filter.MissingImageReaderException;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.graphics.PDXObject;
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject;
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject;
import org.apache.pdfbox.contentstream.operator.Operator;
import org.apache.pdfbox.contentstream.PDFStreamEngine;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.zip.*;
import javax.activation.MimetypesFileTypeMap;
import javax.imageio.IIOException;
import javax.imageio.ImageIO;

public class Extractor extends PDFStreamEngine{
	
	public Extractor() {
	}
	
	static String array[]=null;
	static Map<String, Integer> sinistri= new TreeMap<String,Integer>();
	static Map<String, Integer> sinistriProb= new TreeMap<String,Integer>();
	static List<String> sinistriCrash = new ArrayList<String>();
	
	
	public static int immaginiEstratte = 0;
	public static int immaginiScartate = 0;
	public static int immaginiErrore=0;
	public static int fileCrash=0;
	public static int numSinistriAnalizzati=0;
	
	public static int numPdf=0;
	public static int numZip=0;
	public static int numJpg=0;
	public static int altriFile=0;
	public static int numPptx=0;
	
	public static String dest=null;
	
	public void estrai(String sorgente, String destinazione) throws IOException{
	dest = destinazione;
		
	File scarto= new File(destinazione+"\\immaginiScartate");
		scarto.mkdirs();
	File processati = new File(destinazione+"\\FileProcessati");
		processati.mkdirs();
	String fileName = sorgente;
	File f = new File(fileName);
		
	for(File file : f.listFiles()) {
		//String mimeType = new MimetypesFileTypeMap().getContentType( file );
		String nome= file.getName();
		array=nome.split("_");
		
		//System.out.println(nome);
			
		if(array.length>2) {
		if(array[2].length()>2) {
			array[2]="06";
		}
		}
			
		try {
			
			if (//mimeType != null && mimeType.split("/")[0].equals("image")
					nome.contains(".jpg")||nome.contains(".JPG")||
					nome.contains(".jpeg")||nome.contains(".JPEG")) {
				numJpg++;
				int i=0;
				if(sinistri.containsKey(array[0])) {
					i=sinistri.get(array[0]);
					i++;
					sinistri.put(array[0], i);
				}else {
					i=1;
					sinistri.put(array[0], i);
				}
				
				File immagine = new File(array[0]+"_"+array[1]+"_"+array[2]+"_"+i+".jpg");
					
				BufferedImage image = ImageIO.read(file);
				ImageIO.write(image,"JPG",immagine);
				
				File cartella = new File(destinazione+"\\"+array[0]);
				cartella.mkdirs();
				immagine.renameTo(new File(cartella,immagine.getName()));
					
				immaginiEstratte++;	
				}
			
			else if(nome.contains(".pdf")||nome.contains(".PDF")){
				PDDocument pdf = null;
				numPdf++;
				try{
				pdf = PDDocument.load(file);
				Extractor printer = new Extractor();
				File cartella = new File(destinazione+"\\"+array[0]);
				cartella.mkdirs();
				for( PDPage page : pdf.getPages() ){
					printer.processPage(page);
					}
				
				}
				finally {
				pdf.close();
				}
					
				}
			else if(nome.contains(".zip")||nome.contains(".ZIP")) {
				numZip++;
				unZipIt(file.getAbsolutePath(),destinazione+"\\"+array[0]);
				}
			else if(nome.contains(".pptx")) {
				numPptx++;
				unZipIt(file.getAbsolutePath(),destinazione+"\\"+array[0]);
				}
			else { 
				altriFile++;
				sinistriCrash.add(array[0]);
				fileCrash++;
			}
			
			}catch(Exception e) {
				sinistriCrash.add(array[0]);
				fileCrash++;
			}
			finally {
				file.renameTo(new File(processati,file.getName()));
			}

			}
			
			String path = newFile();
			writeFile(path);
	}

	@Override
	protected void processOperator( Operator operator, List<COSBase> operands) throws IOException{
		try {
			int k=0;
			if(sinistri.containsKey(array[0])) {
				k=sinistri.get(array[0]);	
			}else {
				k=0;
				sinistri.put(array[0], k);
			}
		String operation = operator.getName();
		if( "Do".equals(operation) ){
			COSName objectName = (COSName) operands.get( 0 );
			PDXObject xobject = getResources().getXObject( objectName );
			if( xobject instanceof PDImageXObject){
				PDImageXObject image = (PDImageXObject)xobject;
				int imageWidth = image.getWidth();
				int imageHeight = image.getHeight();
				BufferedImage bImage = new BufferedImage(imageWidth,imageHeight,BufferedImage.TYPE_INT_ARGB);
				k++;
				sinistri.put(array[0], k);
				bImage = image.getImage();
				File immagine = new File(array[0]+"_"+array[1]+"_"+array[2]+"_"+k+".png");
				ImageIO.write(bImage,"PNG",immagine);
				
				if(imageWidth>3*imageHeight || imageHeight>3*imageWidth || imageHeight<=30 || 
						imageWidth<=30) {
					
					immagine.renameTo(new File(dest+"\\immaginiScartate",immagine.getName()));
					immaginiScartate++;
				}
				else {
					immagine.renameTo(new File(dest+"\\"+array[0],immagine.getName()));
				}	
				immaginiEstratte++;
			}
			else if(xobject instanceof PDFormXObject){
				PDFormXObject form = (PDFormXObject)xobject;
				showForm(form);
			}
		}
		else{
			super.processOperator( operator, operands);
		}
		}catch (MissingImageReaderException e) {
			if(sinistriProb.containsKey(array[0])) {
				int j=sinistriProb.get(array[0]);
				j++;
				sinistriProb.put(array[0], j);
			}else {
				int j=1;
				sinistriProb.put(array[0], j);
			}
			immaginiErrore++;
		}
	}
	
	public static void unZipIt(String zipFile, String outputFolder){
		
		byte[] buffer = new byte[1024];
	    int i=0;
		if(sinistri.containsKey(array[0])) {
			i=sinistri.get(array[0]);
		}else {
			i=0;
			sinistri.put(array[0], i);
		}
	     try{

	    	File folder = new File(dest+"\\"+array[0]);
	    	if(!folder.exists()){
	    		folder.mkdir();
	    	}
	
	    	ZipInputStream zis = new ZipInputStream(new FileInputStream(zipFile));
	    	ZipEntry ze = zis.getNextEntry();
	    	while(ze!=null){
	    	if(ze.getName().contains(".jpeg") || ze.getName().contains(".JPEG")
	    				|| ze.getName().contains(".JPG") || ze.getName().contains(".jpg")) {
	    		i++;
	    		sinistri.put(array[0], i);
	    		String fileName = array[0]+"_"+array[1]+"_"+array[2]+"_"+i+".jpg";
	           File newFile = new File(outputFolder + File.separator + fileName);
	           immaginiEstratte++;

	           //Crea le cartelle che non esistono
	            new File(newFile.getParent()).mkdirs();
	            FileOutputStream fos = new FileOutputStream(newFile);
	            int len;
	            while ((len = zis.read(buffer)) > 0) {
	       		fos.write(buffer, 0, len);
	            }
	            fos.close();
	    	}
	    		ze = zis.getNextEntry();
	    		}
	        zis.closeEntry();
	    	zis.close();
	    }catch(IOException ex){
	       ex.printStackTrace();
	    }
	   }
	
	public static String newFile() {
		String path = dest+"\\report.csv";

		try {
			File file = new File(path);
			
			if (file.exists())
				System.out.println("Il file " + path + " esiste");
			else if (file.createNewFile())
				System.out.println("Il file " + path + " è stato creato");
			else
				System.out.println("Il file " + path + " non può essere creato");
		} catch (IOException e) {
			e.printStackTrace();
		}
		return path;
	}

	
	public static void writeFile(String path) { 
	    try {
	        File file = new File(path);
	        //String nomeFile = path;
	        FileReader fr = new FileReader(path);
	        BufferedReader br = new BufferedReader(fr);
	        FileWriter fw = new FileWriter(path, true);
	        //FileWriter fw = new FileWriter(file);
	       
	        //Controlla direttamente dalle cartelle nella cartella di destinazione quanti sinistri non hanno immagini
	        List<String> sinistriNoImmagini = new ArrayList<String>();
	        Map<String, Integer> sinistriImmagini = new TreeMap<String,Integer>();
	        
	        File cartellaDestinazione = new File(dest);
	        for(File f : cartellaDestinazione.listFiles()) {
	        	if(f.isDirectory()) {
	        		int count = f.listFiles().length;
	        		if(count==0) {
	        		sinistriNoImmagini.add(f.getName());
	        		}
	        		else {
	        			sinistriImmagini.put(f.getName(),count);
	        		}
	        	}
	        }
	        
//	        String elencoSinistri="Sinistri analizzati;Immagini\n";
//			for(String s : sinistri.keySet()) {
//				int imm = sinistri.get(s);
//				elencoSinistri+=s+";"+imm+"\n";
//			}
				
			String elencoProblematici="Sinistri problematici\n";
			for(String s: sinistriProb.keySet()) {
				int imm = sinistriProb.get(s);
				elencoProblematici+=s+";"+imm+"\n";
			}
			
			String elencoCrash="Estrazioni fallite\n";
			for(String s: sinistriCrash) {
				elencoCrash+=s+"\n";
			}
			
			String elencoNoImmagini="Sinistri senza immagini\n";
			for(String s: sinistriNoImmagini) {
				if(s.compareTo("immaginiScartate")!=0 && s.compareTo("FileProcessati")!=0)
				elencoNoImmagini+=s+"\n";
			}
			
			String elencoImmagini="Sinistri analizzati;Immagini\n";
			for(String s: sinistriImmagini.keySet()) {
				if(s.compareTo("immaginiScartate")!=0 && s.compareTo("FileProcessati")!=0) {
					int imm = sinistriImmagini.get(s);
					elencoImmagini+=s+";"+imm+"\n";
				}
			}
			
			Map<String,Integer> totaleSinistriAnalizzati = new TreeMap<String,Integer>();
			totaleSinistriAnalizzati.putAll(sinistri);
			totaleSinistriAnalizzati.putAll(sinistriProb);
			
			GregorianCalendar gc = new GregorianCalendar();
			int anno = gc.get(Calendar.YEAR);
			int mese = gc.get(Calendar.MONTH) + 1;
			int giorno = gc.get(Calendar.DATE);
			int ore = gc.get(Calendar.HOUR);
			int min = gc.get(Calendar.MINUTE);
			int sec = gc.get(Calendar.SECOND);
			
			String res = "Estrazione iniziata il "+anno+"/"+mese+"/"+giorno+", alle ore "+ore+":"+min+":"+sec+"\n"
					+ "PDF;"+numPdf+"\n"+ 
								"ZIP;"+numZip+"\n"+ 
								"JPG;"+numJpg+"\n"+ 
								"PPTX;"+numPptx+"\n"+
								"Altri file;"+altriFile+"\n\n"+
								"TotaleImmaginiEstratte;"+immaginiEstratte+"\n"+
								 "ImmaginiScartate;"+immaginiScartate+"\n"+
								"ImmaginiErrore;"+immaginiErrore+"\n"+
								 "Sinistri analizzati;"+totaleSinistriAnalizzati.size()+"\n"+
								"Sinistri senza immagini;"+sinistriNoImmagini.size()+"\n"+
								"Estrazioni fallite;"+sinistriCrash.size()+"\n\n"+
								//elencoSinistri+"\n"+
								elencoImmagini+"\n"+
								elencoProblematici+"\n"+elencoCrash+"\n"+
								elencoNoImmagini+"\n\n";
	        
	        fw.write(res);
	        fw.flush();
	        fw.close();
	    }
	    catch(IOException e) {
	        e.printStackTrace();
	    }
	}	

}
 
