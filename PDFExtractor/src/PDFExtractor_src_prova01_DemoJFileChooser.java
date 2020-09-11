package prova01;


import javax.swing.*;
import java.awt.event.*;
import java.io.File;
import java.io.IOException;
import java.awt.*;
import java.util.*;


public class DemoJFileChooser extends JPanel
   implements ActionListener {
	
 Extractor s = new Extractor();

   JButton sorgente;
   JButton estrai;
   JButton dest;
   JTextField pathDest;
   JTextField pathSorg;
   JButton scegli;
   JFileChooser chooser;
   
   String choosertitle;
   
   JLabel etSorg;
   JLabel etDest;
   JLabel etEstrai;
   JLabel etScegli;
   
   String pathSorgente=null;
   String pathDestinazione=null;

  public DemoJFileChooser() {
	  
	  this.setSize(100,50);
	  this.setLayout(new GridLayout(8,2));
	  
	 JPanel p0 = new JPanel();
	 this.add(p0);
	 etSorg = new JLabel("Inserisci qui il path della sorgente");
	 p0.add(etSorg);
	  
	 JPanel p1 = new JPanel();
	 this.add(p1);
	 pathSorg = new JTextField(20);
	p1.add(pathSorg);
	 sorgente = new JButton("Scegli sorgente");
	 sorgente.addActionListener(this);
	 p1.add(sorgente);
	 
	 JPanel p2b = new JPanel();
	 this.add(p2b);
	 etDest = new JLabel("Inserisci qui il path della destinazione");
	 p2b.add(etDest);
	  
	 JPanel p2 = new JPanel();
		 this.add(p2);
		 pathDest = new JTextField(20);
		 p2.add(pathDest);
		 dest = new JButton("Scegli destinazione");
		 dest.addActionListener(this);
		 p2.add(dest);
		 
		 JPanel p3b = new JPanel();
		 this.add(p3b);
		 etEstrai = new JLabel("Clicca qui per avviare l'estrazione");
		 p3b.add(etEstrai);
		 
	JPanel p3 = new JPanel();
	this.add(p3);
		 estrai = new JButton("Estrai");
		 estrai.addActionListener(this);
		 p3.add(estrai);
		 
		 JPanel p4b = new JPanel();
		 this.add(p4b);
		 etScegli = new JLabel("Clicca qui per scegliere il sinistro di cui visualizzare le immagini");
		 p4b.add(etScegli);
		
	JPanel p4 = new JPanel();
	this.add(p4);
	scegli = new JButton("Scegli sinistro");
    scegli.addActionListener(this);
    p4.add(scegli);
   }
  
  public void choose() {
	  chooser = new JFileChooser(); 
	    //Mostra per prima la cartella che ti indico 
	    chooser.setCurrentDirectory(new java.io.File("."));
	    //Titolo della finestrella
	    chooser.setDialogTitle(choosertitle);
	    //Posso scegliere solo tra le cartelle
	    chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
	  
	    chooser.setAcceptAllFileFilterUsed(false);
  }

  public void actionPerformed(ActionEvent e) { 
	  if(e.getSource()== this.sorgente) {
		  choose();
    if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) { 
    	
	    pathSorgente = chooser.getSelectedFile().getPath();
	    pathSorg.setText(pathSorgente);
      }
	  }
    
    else if(e.getSource() == this.dest) {
      this.choose();
      if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) { 
      	
  	    pathDestinazione = chooser.getSelectedFile().getPath();
  	    pathDest.setText(pathDestinazione);
        }
      }
    else if(e.getSource()== this.estrai) {
    	try {
    		if(pathSorgente==null || pathDestinazione==null) {
    			JOptionPane.showMessageDialog(null,"Devi inserire un path");
    		}
    		else if(pathSorgente.compareTo(pathDestinazione)==0) {
    			JOptionPane.showMessageDialog(null,"Sorgente e destinazione devono essere diverse");
    		}
    		else {
			s.estrai(pathSorg.getText(), pathDest.getText());
    		}
		} catch (IOException e1) {
			
		}
    }
    else if(e.getSource() == this.scegli) {
    	this.choose();
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) { 
        	String pathImmagini = chooser.getSelectedFile().getPath();
      	    File f = new File(pathImmagini);
      	    
      	    	//String path = file.getAbsolutePath();
      	    	try {
      	    		Desktop.getDesktop().open(f);
      	    	}catch(IOException ioe){
      	    		System.out.println("IOEXC");
      	    	}
      	    }
          
        }
     }

//  public Dimension getPreferredSize(){
//    return new Dimension(200, 200);
//    }

  public static void main(String s[]) {
    JFrame frame = new JFrame("Estrattore");
    DemoJFileChooser panel = new DemoJFileChooser();
    frame.addWindowListener(
      new WindowAdapter() {
        public void windowClosing(WindowEvent e) {
          System.exit(0);
          }
        }
      );
    frame.getContentPane().add(panel,"Center");
    //frame.setSize(panel.getPreferredSize());
    frame.setSize(500, 300);
    frame.setVisible(true);
   
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }
	

}

