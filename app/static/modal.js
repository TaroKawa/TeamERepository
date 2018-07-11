
$(document).ready(function () {
    $("#modal_open").click(function () {
    $(this).blur();
    console.log($(".image_created").attr("src"));
    if ($(".image_created").attr("src").match(/赤/)) {
      var wid = 1
    } else if ($(".image_created").attr("src").match(/青/)) {
      var wid = 2
    } else if ($(".image_created").attr("src").match(/黒/)) {
      var wid = 8
    } else {
      var wid = 0
    }

    $("#modal-overlay").fadeIn("slow");

    $("#modal" + wid).fadeIn("slow");
    var which = ".modal-content"
    centeringModalSyncer(which);
  })

  $(".modal-close").unbind().click(function(){
    if ($(".image_created").attr("src").match(/赤/)) {
      var wid = 1
    } else if ($(".image_created").attr("src").match(/青/)) {
      var wid = 2
    } else if ($(".image_created").attr("src").match(/黒/)) {
      var wid = 8
    } else {
      var wid = 0
    }

		$("#modal"+wid+ ", #modal-overlay").fadeOut("slow", function(){
			$("#modal-overlay").removeClass("modal-close");
		});
	});
});

function centeringModalSyncer(which) {

  //画面(ウィンドウ)の幅、高さを取得
  var w = $( window ).width() ;
  var h = $( window ).height() ;

  // コンテンツ(#modal-content)の幅、高さを取得
  // jQueryのバージョンによっては、引数[{margin:true}]を指定した時、不具合を起こします。
  var cw = $( which ).outerWidth( {margin:true} );
  var ch = $( which ).outerHeight( {margin:true} );
  var cw = $( which ).outerWidth();
  var ch = $( which ).outerHeight();

  //センタリングを実行する
  $( which ).css( {"left": ((w - cw)/2) + "px","top": ((h - ch)/2) + "px"} ) ;

 };
